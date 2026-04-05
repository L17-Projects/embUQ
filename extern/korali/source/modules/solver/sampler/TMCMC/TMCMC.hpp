// Vendored from BrieucB/UQ_DPD (vega/gpu-batching)

/** \namespace sampler
* @brief Namespace declaration for modules of type: sampler.
*/

/** \file
* @brief Header file for module: TMCMC.
*/

/** \dir solver/sampler/TMCMC
* @brief Contains code, documentation, and scripts for module: TMCMC.
*/

#pragma once

#include "modules/distribution/distribution.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/distribution/specific/multinomial/multinomial.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include <gsl/gsl_vector.h>

namespace korali
{
namespace solver
{
namespace sampler
{
;

typedef struct fparam_s
{
  const double *loglike;
  size_t Ns;
  double exponent;
  double cov;
} fparam_t;

class TMCMC : public Sampler
{
  public:
   std::string _version;
   size_t _populationSize;
   size_t _maxChainLength;
   size_t _burnIn;
   std::vector<size_t> _perGenerationBurnIn;
   double _targetCoefficientOfVariation;
   double _covarianceScaling;
   double _minAnnealingExponentUpdate;
   double _maxAnnealingExponentUpdate;
   double _stepSize;
   double _domainExtensionFactor;
   korali::distribution::specific::Multinomial* _multinomialGenerator;
   korali::distribution::multivariate::Normal* _multivariateGenerator;
   korali::distribution::univariate::Uniform* _uniformGenerator;
   size_t _currentBurnIn;
   std::vector<int> _chainPendingEvaluation;
   std::vector<int> _chainPendingGradient;
   std::vector<std::vector<double>> _chainCandidates;
   std::vector<double> _chainCandidatesLogLikelihoods;
   std::vector<double> _chainCandidatesLogPriors;
   std::vector<std::vector<double>> _chainCandidatesGradients;
   std::vector<int> _chainCandidatesErrors;
   std::vector<std::vector<double>> _chainCandidatesCovariance;
   std::vector<std::vector<double>> _chainLeaders;
   std::vector<double> _chainLeadersLogLikelihoods;
   std::vector<double> _chainLeadersLogPriors;
   std::vector<std::vector<double>> _chainLeadersGradients;
   std::vector<int> _chainLeadersErrors;
   std::vector<std::vector<double>> _chainLeadersCovariance;
   size_t _finishedChainsCount;
   std::vector<size_t> _currentChainStep;
   std::vector<size_t> _chainLengths;
   double _coefficientOfVariation;
   size_t _chainCount;
   double _annealingExponent;
   double _previousAnnealingExponent;
   size_t _numFinitePriorEvaluations;
   size_t _numFiniteLikelihoodEvaluations;
   size_t _acceptedSamplesCount;
   double _currentAccumulatedLogEvidence;
   double _proposalsAcceptanceRate;
   double _selectionAcceptanceRate;
   std::vector<double> _covarianceMatrix;
   double _maxLoglikelihood;
   std::vector<double> _meanTheta;
   std::vector<std::vector<double>> _sampleDatabase;
   std::vector<double> _sampleLogLikelihoodDatabase;
   std::vector<double> _sampleLogPriorDatabase;
   std::vector<std::vector<double>> _sampleGradientDatabase;
   std::vector<int> _sampleErrorDatabase;
   std::vector<std::vector<double>> _sampleCovarianceDatabase;
   std::vector<double> _upperExtendedBoundaries;
   std::vector<double> _lowerExtendedBoundaries;
   size_t _numLUDecompositionFailuresProposal;
   size_t _numEigenDecompositionFailuresProposal;
   size_t _numInversionFailuresProposal;
   size_t _numNegativeDefiniteProposals;
   size_t _numCholeskyDecompositionFailuresProposal;
   size_t _numCovarianceCorrections;
   double _targetAnnealingExponent;

  bool checkTermination() override;
  void getConfiguration(knlohmann::json& js) override;
  void setConfiguration(knlohmann::json& js) override;
  void applyModuleDefaults(knlohmann::json& js) override;
  void applyVariableDefaults() override;
  void setBurnIn();
  void prepareGeneration();
  void processGeneration();
  void minSearch(double const *fj, size_t fn, double pj, double objTol, double &xmin, double &fmin);
  void processCandidate(const size_t sampleId);
  void calculateGradients(std::vector<Sample> &samples);
  void calculateProposals(std::vector<Sample> &samples);
  void generateCandidate(const size_t sampleId);
  void updateDatabase(const size_t sampleId);
  double calculateAcceptanceProbability(const size_t sampleId);
  static double calculateSquaredCVDifference(double x, const double *loglike, size_t Ns, double exponent, double targetCV);
  static double calculateSquaredCVDifferenceOptimizationWrapper(const gsl_vector *v, void *param);
  size_t N;
  void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
  void finalize() override;
};

} //sampler
} //solver
} //korali
;
