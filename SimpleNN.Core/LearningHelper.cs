using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleNN.Core
{
    public static class LearningHelper
    {
        public static void ProcessGradients(this IEnumerable<IGradientLayerInfo> infos, double[] outputs/*, bool resetValues = true*/)
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
                        //if (resetValues)
                        //{
                        //    //info.Errors[j] = 0.0;
                        //    info.TotalGradients[j] = 0.0;
                        //    info.Gradients[j] = 0.0;
                        //}
                        var curError = outputs[j] - value;
                        //info.Errors[j] += curError;
                        var gradient = layer.Derivate(value) * curError;
                        info.Gradients[j] = gradient;
                        //info.TotalGradients[j] += gradient;
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
                        //if (resetValues)
                        //{
                        //    //info.Errors[j] = 0.0;
                        //    info.TotalGradients[j] = 0.0;
                        //}
                        //info.Errors[j] += sum;
                        var gradient = layer.Derivate(value) * sum;
                        info.Gradients[j] = gradient;
                        //info.TotalGradients[j] += gradient;
                    }
                }
                prev = info;
            }
        }

        public static void InitNeurons(this IEnumerable<IGradientLayerInfo> layerInfos, Random rnd)
        {
            foreach (var info in layerInfos)
            {
                var layer = info.Layer;
                foreach (var neuron in layer)
                {
                    neuron.Bias = rnd.NextDouble() - 0.5;
                    for (int i = 0; i < neuron.Weights.Length; i++) neuron.Weights[i] = rnd.NextDouble() - 0.5;
                }
            }
        }

        public static void InitNG(this IEnumerable<IGradientLayerInfo> layerInfos, NeuronNet neuronNet, Random rnd)
        {
            if (neuronNet.Count < 2) return;
            int n = neuronNet[0].Inputs.Length, p = 0;
            for (int index = 0; index < (neuronNet.Count - 1); index++) p += neuronNet[index].Outputs.Length;
            double beta = 0.7*Math.Pow(p, 1.0/n);

            foreach (var info in layerInfos)
            {
                var layer = info.Layer;
                foreach (var neuron in layer)
                {
                    neuron.Bias = (2.0*rnd.NextDouble() - 1.0)*beta;
                    var mod = Math.Sqrt(neuron.Weights.Sum(x => x*x));
                    for (int i = 0; i < neuron.Weights.Length; i++) neuron.Weights[i] = beta*neuron.Weights[i]/mod;
                }
            }
        }

        public static void ProcessDeltas(this IEnumerable<IGradientLayerInfo> layerInfos, double eta, double alpha, bool resetDelta = true)
        {
            foreach (var info in layerInfos)
            {
                var layer = info.Layer;
                for (int neuronIndex = 0; neuronIndex < layer.Count; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];
                    var neuronKoef1 = eta*info.Gradients[neuronIndex]*(1.0 - alpha);
                    for (int weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        var currentDelta = neuronKoef1 * layer.Inputs[weightIndex];
                        if (resetDelta)
                            info.Deltas[neuronIndex, weightIndex] = currentDelta;
                        else
                            info.Deltas[neuronIndex, weightIndex] += currentDelta;
                    }
                    var biasDelta = neuronKoef1*1.0;
                    if (resetDelta)
                        info.BiasDeltas[neuronIndex] = biasDelta;
                    else
                        info.BiasDeltas[neuronIndex] += biasDelta;
                }
            }
        }
    }
}