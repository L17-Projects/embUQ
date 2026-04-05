#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "modules/distribution/distribution.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

class Psi : public Hierarchical
{
  private:
  const double _log2Pi = 1.83787706640934533908193770912476;

  struct conditionalPriorInfo
  {
    std::vector<size_t> _samplePositions;
    std::vector<double *> _samplePointers;
  };

  struct nativeParameterSource
  {
    bool _isVariable = false;
    size_t _position = 0;
    double _value = 0.0;
  };

  enum nativeConditionalPriorKind { Normal, Uniform };

  struct nativeConditionalPriorSpec
  {
    nativeConditionalPriorKind _kind = nativeConditionalPriorKind::Normal;
    size_t _sampleDimension = 0;
    nativeParameterSource _parameterA;
    nativeParameterSource _parameterB;
    bool isConstant() const { return _parameterA._isVariable == false && _parameterB._isVariable == false; }
  };

  struct nativeSubProblemCache
  {
    std::vector<std::vector<double>> _sampleCoordinatesByVariable;
    std::vector<double> _baseLogWeights;
  };

  size_t _subProblemsCount = 0;
  size_t _subProblemsVariablesCount = 0;
  std::vector<std::vector<std::vector<double>>> _subProblemsSampleCoordinates;
  std::vector<std::vector<double>> _subProblemsSampleLogLikelihoods;
  std::vector<std::vector<double>> _subProblemsSampleLogPriors;
  std::vector<conditionalPriorInfo> _conditionalPriorInfos;
  std::vector<nativeConditionalPriorSpec> _nativeConditionalPriorSpecs;
  std::vector<size_t> _nativeDynamicConditionalPriorIndexes;
  std::vector<nativeSubProblemCache> _nativeSubProblemCaches;

  size_t findVariablePosition(const std::string &variableName, const std::string &propertyName, size_t priorIndex) const;
  nativeParameterSource createNativeParameterSource(const std::string &variableName, double value, const std::string &propertyName, size_t priorIndex) const;
  double resolveNativeParameter(const nativeParameterSource &source, const std::vector<double> &parameters) const;
  double evaluateNativeConditionalLogDensity(const nativeConditionalPriorSpec &spec, const std::vector<double> &parameters, double sampleValue) const;
  void initializeNativeBatchCache();
  void initializeNativeCudaBatch();
  void releaseNativeCudaBatch();
  void evaluateBatchNativeCpu(korali::Sample &sample);
  void evaluateBatchNativeCuda(korali::Sample &sample);
  bool usesExternalBatchBackend() const;
  bool usesNativeCpuBatchBackend() const;
  bool usesNativeCudaBatchBackend() const;

  public:
   int _useBatchEvaluation = 0;
   std::string _batchEvaluationBackend = "External";
   std::vector<knlohmann::json> _subExperiments;
   std::vector<std::string> _conditionalPriors;
   std::uint64_t _batchComputationalModel = 0;

  void getConfiguration(knlohmann::json& js) override;
  void setConfiguration(knlohmann::json& js) override;
  void applyModuleDefaults(knlohmann::json& js) override;
  void applyVariableDefaults() override;
  ~Psi() override;

  bool _hasBatchComputationalModel = false;
  std::vector<size_t> _conditionalPriorIndexes;
  void updateConditionalPriors(korali::Sample &sample);
  void evaluateLogLikelihood(korali::Sample &sample) override;
  void evaluateBatch(korali::Sample &sample) override;
  bool supportsEvaluateBatch() const override;
  void initialize() override;
};

} //hierarchical
} //problem
} //korali
;
