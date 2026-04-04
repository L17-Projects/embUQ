/** \namespace hierarchical
* @brief Namespace declaration for modules of type: hierarchical.
*/

/** \file
* @brief Header file for module: Theta.
*/

/** \dir problem/hierarchical/theta
* @brief Contains code, documentation, and scripts for module: Theta.
*/

#pragma once

#include "modules/problem/bayesian/bayesian.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

class Theta : public Hierarchical
{
  private:
  korali::Experiment _psiExperimentObject;
  korali::Experiment _subExperimentObject;
  size_t _psiVariableCount;
  size_t _psiProblemSampleCount;
  std::vector<std::vector<double>> _psiProblemSampleCoordinates;
  std::vector<double> _psiProblemSampleLogLikelihoods;
  std::vector<double> _psiProblemSampleLogPriors;
  korali::problem::hierarchical::Psi *_psiProblem;
  size_t _subProblemVariableCount;
  size_t _subProblemSampleCount;
  std::vector<std::vector<double>> _subProblemSampleCoordinates;
  std::vector<double> _subProblemSampleLogLikelihoods;
  std::vector<double> _subProblemSampleLogPriors;
  std::vector<double> _precomputedLogDenominator;

  double calculateHierarchicalCorrection(const std::vector<double> &parameters);
  std::vector<double> calculateHierarchicalCorrectionBatch(const std::vector<std::vector<double>> &batchParameters);

  public:
   knlohmann::json _subExperiment;
   knlohmann::json _psiExperiment;

  void getConfiguration(knlohmann::json& js) override;
  void setConfiguration(knlohmann::json& js) override;
  void applyModuleDefaults(knlohmann::json& js) override;
  void applyVariableDefaults() override;
  void evaluateLogLikelihood(korali::Sample &sample) override;
  void evaluateBatch(korali::Sample &sample) override;
  bool supportsEvaluateBatch() const override;
  void initialize() override;
};

} //hierarchical
} //problem
} //korali
;
