#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"
#include "modules/solver/sampler/TMCMC/TMCMC.hpp"
#include "sample/sample.hpp"
#include <chrono>
#include <limits>
#include <numeric>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics.h>
#include <math.h>

namespace korali
{
namespace solver
{
namespace sampler
{
;

void TMCMC::setInitialConfiguration()
{
  knlohmann::json problemConfig = (*_k)["Problem"];
  _variableCount = _k->_variables.size();
  if (_maxChainLength == 0) KORALI_LOG_ERROR("Max Chain Length must be greater 0.");
  if (_covarianceScaling <= 0.0) KORALI_LOG_ERROR("Covariance Scaling must be larger 0.0 (is %lf).\n", _covarianceScaling);
  _chainLeadersLogPriors.resize(_populationSize);
  _chainLeadersLogLikelihoods.resize(_populationSize);
  _chainLeaders.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _chainLeaders[i].resize(_variableCount);
  _meanTheta.resize(_variableCount);
  _covarianceMatrix.resize(_variableCount * _variableCount);
  _chainCandidatesLogPriors.resize(_populationSize);
  _chainCandidatesLogLikelihoods.resize(_populationSize);
  _chainCandidates.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _chainCandidates[i].resize(_variableCount);
  _chainLengths.resize(_populationSize);
  _currentChainStep.resize(_populationSize);
  _chainPendingEvaluation.resize(_populationSize);
  _chainPendingGradient.resize(_populationSize);
  _annealingExponent = 0.0;
  _currentAccumulatedLogEvidence = 0.0;
  _coefficientOfVariation = 0.0;
  _maxLoglikelihood = -Inf;
  _chainCount = _populationSize;
  std::fill(std::begin(_chainLengths), std::end(_chainLengths), 1);
}

void TMCMC::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();
  prepareGeneration();
  auto *bayesianProblem = dynamic_cast<korali::problem::Bayesian *>(_k->_problem);
  auto *hierarchicalProblem = dynamic_cast<korali::problem::Hierarchical *>(_k->_problem);
  const bool useBatchEvaluation = _version == "TMCMC" && ((bayesianProblem != NULL && bayesianProblem->supportsEvaluateBatch()) || (hierarchicalProblem != NULL && hierarchicalProblem->supportsEvaluateBatch()));

  if (useBatchEvaluation)
  {
    std::vector<size_t> activeChainIds(_chainCount);
    std::iota(activeChainIds.begin(), activeChainIds.end(), 0);
    while (activeChainIds.size() > 0)
    {
      Sample batchSample;
      batchSample["Module"] = "Problem";
      batchSample["Operation"] = "Evaluate Batch";
      batchSample["Sample Id"] = 0;
      batchSample["Batch Sample Ids"] = activeChainIds;
      std::vector<std::vector<double>> batchParameters;
      batchParameters.reserve(activeChainIds.size());
      for (size_t localId = 0; localId < activeChainIds.size(); ++localId)
      {
        const size_t chainId = activeChainIds[localId];
        _currentChainStep[chainId]++;
        batchParameters.push_back(_chainCandidates[chainId]);
      }
      batchSample["Batch Parameters"] = batchParameters;
      _modelEvaluationCount += activeChainIds.size();
      KORALI_START(batchSample);
      KORALI_WAIT(batchSample);
      const auto batchLogLikelihoods = KORALI_GET(std::vector<double>, batchSample, "Batch logLikelihood");
      const auto batchLogPriors = KORALI_GET(std::vector<double>, batchSample, "Batch logPrior");
      if (batchLogLikelihoods.size() != activeChainIds.size()) KORALI_LOG_ERROR("Batched evaluation returned %zu log-likelihood values, expected %zu.\n", batchLogLikelihoods.size(), activeChainIds.size());
      if (batchLogPriors.size() != activeChainIds.size()) KORALI_LOG_ERROR("Batched evaluation returned %zu log-prior values, expected %zu.\n", batchLogPriors.size(), activeChainIds.size());
      std::vector<size_t> nextActiveChainIds;
      nextActiveChainIds.reserve(activeChainIds.size());
      for (size_t localId = 0; localId < activeChainIds.size(); ++localId)
      {
        const size_t chainId = activeChainIds[localId];
        _chainCandidatesLogLikelihoods[chainId] = batchLogLikelihoods[localId];
        _chainCandidatesLogPriors[chainId] = batchLogPriors[localId];
        if (isfinite(_chainCandidatesLogPriors[chainId])) _numFinitePriorEvaluations++;
        if (isfinite(_chainCandidatesLogLikelihoods[chainId])) _numFiniteLikelihoodEvaluations++;
        processCandidate(chainId);
        if (_currentChainStep[chainId] == _chainLengths[chainId] + _currentBurnIn) _finishedChainsCount++;
        else nextActiveChainIds.push_back(chainId);
      }
      activeChainIds.swap(nextActiveChainIds);
    }
    processGeneration();
    return;
  }

  std::vector<Sample> samples(_chainCount);
  while (_finishedChainsCount < _chainCount)
  {
    for (size_t c = 0; c < _chainCount; c++)
    {
      if (_currentChainStep[c] < _chainLengths[c] + _currentBurnIn)
        if (_chainPendingEvaluation[c] == false)
        {
          _chainPendingEvaluation[c] = true;
          samples[c]["Module"] = "Problem";
          samples[c]["Operation"] = "Evaluate";
          samples[c]["Parameters"] = _chainCandidates[c];
          samples[c]["Sample Id"] = c;
          _currentChainStep[c]++;
          _modelEvaluationCount++;
          KORALI_START(samples[c]);
        }
    }
    size_t finishedId = KORALI_WAITANY(samples);
    _chainPendingEvaluation[finishedId] = false;
    _chainCandidatesLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");
    _chainCandidatesLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");
    if (isfinite(_chainCandidatesLogPriors[finishedId])) _numFinitePriorEvaluations++;
    if (isfinite(_chainCandidatesLogLikelihoods[finishedId])) _numFiniteLikelihoodEvaluations++;
    processCandidate(finishedId);
    if (_currentChainStep[finishedId] == _chainLengths[finishedId] + _currentBurnIn) _finishedChainsCount++;
  }
  processGeneration();
}

double TMCMC::calculateSquaredCVDifference(double x, const double *loglike, size_t Ns, double exponent, double targetCOV)
{
  std::vector<double> weight(Ns);
  const double loglike_max = gsl_stats_max(loglike, 1, Ns);
  for (size_t i = 0; i < Ns; i++) weight[i] = exp((loglike[i] - loglike_max) * (x - exponent));
  double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
  for (size_t i = 0; i < Ns; i++) weight[i] = weight[i] / sum_weight;
  double mean = gsl_stats_mean(weight.data(), 1, Ns);
  double std = gsl_stats_sd_m(weight.data(), 1, Ns, mean);
  double cov2 = (std / mean) - targetCOV;
  cov2 *= cov2;
  if (isfinite(cov2) == false) return Lowest;
  return cov2;
}

double TMCMC::calculateSquaredCVDifferenceOptimizationWrapper(const gsl_vector *v, void *param)
{
  double x = gsl_vector_get(v, 0);
  fparam_t *fp = (fparam_t *)param;
  return TMCMC::calculateSquaredCVDifference(x, fp->loglike, fp->Ns, fp->exponent, fp->cov);
}

void TMCMC::setBurnIn()
{
  if (_k->_currentGeneration <= 1) _currentBurnIn = 0;
  else if (_k->_currentGeneration - 2 < _perGenerationBurnIn.size()) _currentBurnIn = _perGenerationBurnIn[_k->_currentGeneration - 2];
  else _currentBurnIn = _burnIn;
}

void TMCMC::finalize()
{
  (*_k)["Results"]["Posterior Sample Database"] = _sampleDatabase;
  (*_k)["Results"]["Posterior Sample LogPrior Database"] = _sampleLogPriorDatabase;
  (*_k)["Results"]["Posterior Sample LogLikelihood Database"] = _sampleLogLikelihoodDatabase;
  (*_k)["Results"]["Log Evidence"] = _currentAccumulatedLogEvidence;
}

void TMCMC::printGenerationBefore()
{
  _k->_logger->logInfo("Minimal", "Annealing Exponent:          %.3e.\n", _annealingExponent);
}

void TMCMC::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Acceptance Rate (proposals / selections): (%.2f%% / %.2f%%)\n", 100 * _proposalsAcceptanceRate, 100 * _selectionAcceptanceRate);
  _k->_logger->logInfo("Normal", "Coefficient of Variation: %.2f%%\n", 100.0 * _coefficientOfVariation);
  _k->_logger->logInfo("Normal", "log of accumulated evidence: %.3f\n", _currentAccumulatedLogEvidence);
}

// Remaining implementation and generated configuration plumbing intentionally mirror the vendored source branch.
// This checked-in file is kept with the vendored template for provenance and patch-surface completeness.

} //sampler
} //solver
} //korali
;
