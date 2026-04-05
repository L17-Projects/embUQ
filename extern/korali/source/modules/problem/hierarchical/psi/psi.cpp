#include "modules/conduit/conduit.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"
#include "sample/sample.hpp"

#include <cmath>
#include <limits>

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

bool Psi::usesExternalBatchBackend() const { return _batchEvaluationBackend == "External"; }
bool Psi::usesNativeCpuBatchBackend() const { return _batchEvaluationBackend == "NativeCpu"; }
bool Psi::usesNativeCudaBatchBackend() const { return _batchEvaluationBackend == "NativeCuda"; }

Psi::~Psi() { releaseNativeCudaBatch(); }

size_t Psi::findVariablePosition(const std::string &variableName, const std::string &, size_t) const
{
  for (size_t i = 0; i < _k->_variables.size(); ++i)
    if (_k->_variables[i]->_name == variableName) return i;
  KORALI_LOG_ERROR("Could not resolve variable '%s' while configuring Hierarchical/Psi native batch evaluation.\n", variableName.c_str());
  return 0;
}

Psi::nativeParameterSource Psi::createNativeParameterSource(const std::string &variableName, double value, const std::string &propertyName, size_t priorIndex) const
{
  nativeParameterSource source;
  source._value = value;
  if (variableName.empty() == false)
  {
    source._isVariable = true;
    source._position = findVariablePosition(variableName, propertyName, priorIndex);
  }
  return source;
}

double Psi::resolveNativeParameter(const nativeParameterSource &source, const std::vector<double> &parameters) const
{
  if (source._isVariable == false) return source._value;
  if (source._position >= parameters.size())
    KORALI_LOG_ERROR("Native batch evaluation requested Psi parameter %zu, but the batch sample only contains %zu parameters.\n", source._position, parameters.size());
  return parameters[source._position];
}

double Psi::evaluateNativeConditionalLogDensity(const nativeConditionalPriorSpec &spec, const std::vector<double> &parameters, double sampleValue) const
{
  if (spec._kind == nativeConditionalPriorKind::Normal)
  {
    const double mean = resolveNativeParameter(spec._parameterA, parameters);
    const double standardDeviation = resolveNativeParameter(spec._parameterB, parameters);
    if (standardDeviation <= 0.0) return -Inf;
    const double delta = (sampleValue - mean) / standardDeviation;
    return -0.5 * _log2Pi - std::log(standardDeviation) - 0.5 * delta * delta;
  }

  const double minimum = resolveNativeParameter(spec._parameterA, parameters);
  const double maximum = resolveNativeParameter(spec._parameterB, parameters);
  if (maximum <= minimum) return -Inf;
  if (sampleValue < minimum || sampleValue > maximum) return -Inf;
  return -std::log(maximum - minimum);
}

void Psi::initializeNativeBatchCache() {}
void Psi::initializeNativeCudaBatch() {}
void Psi::releaseNativeCudaBatch() {}

void Psi::initialize()
{
  Hierarchical::initialize();
  _hasBatchComputationalModel = (_batchComputationalModel != 0);
}

void Psi::updateConditionalPriors(Sample &) {}

void Psi::evaluateLogLikelihood(Sample &sample)
{
  sample["logLikelihood"] = -Inf;
}

bool Psi::supportsEvaluateBatch() const
{
  if (_useBatchEvaluation == 0) return false;
  if (usesNativeCpuBatchBackend()) return true;
  if (usesNativeCudaBatchBackend()) return true;
  return usesExternalBatchBackend() && _hasBatchComputationalModel;
}

void Psi::evaluateBatchNativeCpu(Sample &sample)
{
  sample["Batch logPrior"] = std::vector<double>();
  sample["Batch logLikelihood"] = std::vector<double>();
}

void Psi::evaluateBatchNativeCuda(Sample &sample)
{
  sample["Batch logPrior"] = std::vector<double>();
  sample["Batch logLikelihood"] = std::vector<double>();
}

void Psi::evaluateBatch(Sample &sample)
{
  if (_useBatchEvaluation == 0)
    KORALI_LOG_ERROR("Batch evaluation requested, but 'Use Batch Evaluation' is disabled.\n");

  if (usesNativeCpuBatchBackend()) { evaluateBatchNativeCpu(sample); return; }
  if (usesNativeCudaBatchBackend()) { evaluateBatchNativeCuda(sample); return; }

  sample["Batch logPrior"] = std::vector<double>();
  sample["Batch logLikelihood"] = std::vector<double>();
}

void Psi::setConfiguration(knlohmann::json& js)
{
  if (isDefined(js, "Use Batch Evaluation")) { _useBatchEvaluation = js["Use Batch Evaluation"].get<int>(); eraseValue(js, "Use Batch Evaluation"); }
  if (isDefined(js, "Batch Evaluation Backend")) { _batchEvaluationBackend = js["Batch Evaluation Backend"].get<std::string>(); eraseValue(js, "Batch Evaluation Backend"); }
  if (isDefined(js, "Sub Experiments")) { _subExperiments = js["Sub Experiments"].get<std::vector<knlohmann::json>>(); eraseValue(js, "Sub Experiments"); }
  if (isDefined(js, "Conditional Priors")) { _conditionalPriors = js["Conditional Priors"].get<std::vector<std::string>>(); eraseValue(js, "Conditional Priors"); }
  if (isDefined(js, "Batch Computational Model")) { _batchComputationalModel = js["Batch Computational Model"].get<std::uint64_t>(); eraseValue(js, "Batch Computational Model"); }
  Hierarchical::setConfiguration(js);
  _type = "hierarchical/psi";
  if (isDefined(js, "Type")) eraseValue(js, "Type");
}

void Psi::getConfiguration(knlohmann::json& js)
{
  js["Type"] = _type;
  js["Use Batch Evaluation"] = _useBatchEvaluation;
  js["Batch Evaluation Backend"] = _batchEvaluationBackend;
  js["Sub Experiments"] = _subExperiments;
  js["Conditional Priors"] = _conditionalPriors;
  js["Batch Computational Model"] = _batchComputationalModel;
  Hierarchical::getConfiguration(js);
}

void Psi::applyModuleDefaults(knlohmann::json& js)
{
  std::string defaultString = "{\"Use Batch Evaluation\": false, \"Batch Evaluation Backend\": \"External\", \"Batch Computational Model\": 0}";
  knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
  mergeJson(js, defaultJs);
  Hierarchical::applyModuleDefaults(js);
}

void Psi::applyVariableDefaults()
{
  Hierarchical::applyVariableDefaults();
}

} //hierarchical
} //problem
} //korali
;
