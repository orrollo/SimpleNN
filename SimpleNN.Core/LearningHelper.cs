using System.Collections.Generic;

namespace SimpleNN.Core
{
    public static class LearningHelper
    {
        public static void ProcessGradients(this IEnumerable<IGradientLayerInfo> infos, double[] outputs, bool resetValues = true)
        {
            IGradientLayerInfo prev = null;
            foreach (var info in infos)
            {
                var layer = info.Layer;
                var values = layer.Outputs;
                if (prev == null)
                {
                    // output layer
                    for (int j = 0; j < values.Length; j++)
                    {
                        var value = values[j];
                        if (resetValues)
                        {
                            info.Errors[j] = 0.0;
                            info.Gradients[j] = 0.0;
                        }
                        info.Errors[j] += outputs[j] - value;
                        info.Gradients[j] += layer.Derivate(value) * info.Errors[j];
                    }
                }
                else
                {
                    // hidden layers
                    for (int j = 0; j < values.Length; j++)
                    {
                        var value = values[j];
                        double sum = 0.0;
                        for (int k = 0; k < prev.Layer.Count; k++) sum += prev.Layer[k].Weights[j] * prev.Gradients[k];
                        if (resetValues)
                        {
                            info.Errors[j] = 0.0;
                            info.Gradients[j] = 0.0;
                        }
                        info.Errors[j] += sum;
                        info.Gradients[j] += layer.Derivate(value) * info.Errors[j];
                    }
                }
                prev = info;
            }
        }
    }
}