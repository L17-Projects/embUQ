/** \namespace hierarchical
* @brief Namespace declaration for modules of type: hierarchical.
*/

/** \file
* @brief Header file for module: Psi.
*/

/** \dir problem/hierarchical/psi
* @brief Contains code, documentation, and scripts for module: Psi.
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef _KORALI_USE_CUDA_BATCH
  #include <cuda.h>
#endif

#include "modules/distribution/distribution.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

/**
* @brief Class declaration for module: Psi.
*/
class Psi : public Hierarchical
{
  private:
  /**
   * @brief Precomputed constant needed for direct Normal log-density evaluation.
   */
  const double _log2Pi = 1.83787706640934533908193770912476;

  /**
   * @brief Stores the pre-computed positions (pointers) of the conditional priors to evaluate for performance
   */
  struct conditionalPriorInfo
  {
    /**
     * @brief Stores the position of the conditional prior
     */
    std::vector<size_t> _samplePositions;

    /**
     * @brief Stores the pointer of the conditional prior
     */
    std::vector<double *> _samplePointers;
  };

  /**
   * @brief Describes whether a native batch prior parameter is constant or read from the Psi sample.
   */
  struct nativeParameterSource
  {
    bool _isVariable = false;
    size_t _position = 0;
    double _value = 0.0;
  };

  /**
   * @brief Enumerates the conditional priors supported by the native batch backend.
   */
  enum nativeConditionalPriorKind
  {
    Normal,
    Uniform
  };

  /**
   * @brief Stores the metadata required to evaluate one conditional prior natively in batch mode.
   */
  struct nativeConditionalPriorSpec
  {
    nativeConditionalPriorKind _kind = nativeConditionalPriorKind::Normal;
    size_t _sampleDimension = 0;
    nativeParameterSource _parameterA;
    nativeParameterSource _parameterB;

    bool isConstant() const
    {
      return _parameterA._isVariable == false && _parameterB._isVariable == false;
    }
  };

  /**
   * @brief Stores sub-problem data in a structure-of-arrays layout for native batch evaluation.
   */
  struct nativeSubProblemCache
  {
    std::vector<std::vector<double>> _sampleCoordinatesByVariable;
    std::vector<double> _baseLogWeights;
  };

  /**
   * @brief Stores the number of subproblems
   */
  size_t _subProblemsCount;

  /**
   * @brief Stores the number of variables in the subproblems (all must be the same)
   */
  size_t _subProblemsVariablesCount;

  /**
   * @brief Stores the sample coordinates of all the subproblems
   */
  std::vector<std::vector<std::vector<double>>> _subProblemsSampleCoordinates;

  /**
   * @brief Stores the sample logLikelihoods of all the subproblems
   */
  std::vector<std::vector<double>> _subProblemsSampleLogLikelihoods;

  /**
   * @brief Stores the sample logPriors of all the subproblems
   */
  std::vector<std::vector<double>> _subProblemsSampleLogPriors;

  /**
   * @brief Stores the precomputed conditional prior information, for performance
   */
  std::vector<conditionalPriorInfo> _conditionalPriorInfos;

  /**
   * @brief Stores the native conditional-prior descriptors used by the native batch backend.
   */
  std::vector<nativeConditionalPriorSpec> _nativeConditionalPriorSpecs;

  /**
   * @brief Stores the subset of conditional priors that still depend on the current Psi sample.
   */
  std::vector<size_t> _nativeDynamicConditionalPriorIndexes;

  /**
   * @brief Stores the cached sub-problem data used by the native batch backend.
   */
  std::vector<nativeSubProblemCache> _nativeSubProblemCaches;

#ifdef _KORALI_USE_CUDA_BATCH
  bool _nativeCudaInitialized = false;
  int _nativeCudaDeviceId = 0;
  CUdevice _nativeCudaDevice = 0;
  CUcontext _nativeCudaContext = nullptr;
  CUmodule _nativeCudaModule = nullptr;
  CUfunction _nativeCudaLogLikelihoodKernel = nullptr;
  std::vector<CUdeviceptr> _nativeCudaSubProblemCoordinates;
  std::vector<CUdeviceptr> _nativeCudaSubProblemBaseLogWeights;
  CUdeviceptr _nativeCudaSubProblemCoordinatesDevice = 0;
  CUdeviceptr _nativeCudaSubProblemBaseLogWeightsDevice = 0;
  CUdeviceptr _nativeCudaSubProblemSampleCountsDevice = 0;
  CUdeviceptr _nativeCudaPriorKindsDevice = 0;
  CUdeviceptr _nativeCudaParameterAIsVariableDevice = 0;
  CUdeviceptr _nativeCudaParameterAPositionDevice = 0;
  CUdeviceptr _nativeCudaParameterAValueDevice = 0;
  CUdeviceptr _nativeCudaParameterBIsVariableDevice = 0;
  CUdeviceptr _nativeCudaParameterBPositionDevice = 0;
  CUdeviceptr _nativeCudaParameterBValueDevice = 0;
#endif

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
  /**
  * @brief Enables the optional batched likelihood path for Hierarchical/Psi problems.
  */
   int _useBatchEvaluation;
  /**
  * @brief Selects the backend used when batched Hierarchical/Psi evaluation is enabled.
  */
   std::string _batchEvaluationBackend;
  /**
  * @brief Provides results from previous Bayesian Inference sampling experiments.
  */
   std::vector<knlohmann::json> _subExperiments;
  /**
  * @brief List of conditional priors to use in the hierarchical problem.
  */
   std::vector<std::string> _conditionalPriors;
  /**
  * @brief [Internal Use] Optional Python callback used to evaluate a batch of Psi log-likelihoods.
  */
   std::uint64_t _batchComputationalModel;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  ~Psi() override;

  /**
   * @brief Tracks whether a batch computational model was explicitly configured.
   */
  bool _hasBatchComputationalModel = false;

  /**
   * @brief Stores the indexes of conditional priors to Experiment variables
   */
  std::vector<size_t> _conditionalPriorIndexes;

  /**
   * @brief Updates the distribution parameters for the conditional priors, given variable values in the sample.
   * @param sample A Korali Sample
   */
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
