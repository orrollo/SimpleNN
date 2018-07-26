using System;
using System.Collections.Generic;

namespace SimpleNN.Core
{
    public class RPropParams
    {
        
    }

    public class RPropLearning
    {
        private List<LayerInfo> infos;

        class LayerInfo : IGradientLayerInfo
        {
            private object _gradients;

            public LayerInfo(NeuronLayer layer)
            {
                Layer = layer;
                var length = layer.Outputs.Length;
                Gradients = new double[length];
                Deltas = new double[length, layer.Inputs.Length];
                BiasDeltas = new double[length];
                PrevDeltas = new double[length, layer.Inputs.Length];
                PrevBiasDeltas = new double[length];
            }

            public NeuronLayer Layer { get; set; }
            public double[] Gradients { get; set; }

            public double[,] Deltas { get; set; }
            public double[] BiasDeltas { get; set; }

            public double[,] PrevDeltas { get; set; }
            public double[] PrevBiasDeltas { get; set; }
        }

        public RPropLearning(NeuronNet network)
        {
            Network = network;
            // reverse order
            infos = new List<LayerInfo>();
            foreach (var layer in network) infos.Insert(0, new LayerInfo(layer));
        }

        public NeuronNet Network { get; set; }

        public void Train(DataSamples samples, DataSamples testSamples, RPropParams param)
        {
            var rnd = new Random();
            infos.InitNeurons(rnd);

            // init deltas
            ProcessSamples(samples);

            while (true)
            {
                infos.UpdateWeights(0.0);

                var error = Network.Error(testSamples);
                if (error < 0.05) break;

                // exchange arrays
                ExchangeDeltas();
                ProcessSamples(samples);
                // calc changes in direction
                foreach (var layerInfo in infos)
                {
                    var layer = layerInfo.Layer;
                    for (int neuronIndex = 0; neuronIndex < layer.Count; neuronIndex++)
                    {
                        for (int weightIndex = 0; weightIndex < layer.Inputs.Length; weightIndex++)
                        {
                            double s0 = layerInfo.PrevDeltas[neuronIndex, weightIndex];
                            double s1 = layerInfo.Deltas[neuronIndex, weightIndex];
                            double sign = s0 > 0 ? -1.0 : 1.0, value = Math.Abs(layerInfo.PrevDeltas[neuronIndex, weightIndex]);
                            if (s0*s1 > 0.0)
                                value = Math.Min(50, value*1.2);
                            else if (s0*s1 < 0.0)
                                value = -sign*Math.Max(1e-30, value*0.5);
                            else
                                value = 1e-5;
                            layerInfo.Deltas[neuronIndex, weightIndex] = sign*value;
                        }
                    }
                }
            }

        }

        private void ExchangeDeltas()
        {
            foreach (var layerInfo in infos)
            {
                var tmp1 = layerInfo.PrevDeltas;
                layerInfo.PrevDeltas = layerInfo.Deltas;
                layerInfo.Deltas = tmp1;
                var tmp2 = layerInfo.PrevBiasDeltas;
                layerInfo.PrevBiasDeltas = layerInfo.BiasDeltas;
                layerInfo.BiasDeltas = tmp2;
            }
        }

        private void ProcessSamples(DataSamples samples)
        {
            bool first = true;
            foreach (var sample in samples)
            {
                // forward step
                Network.Inputs = sample.Inputs;
                Network.Calc();
                // backward
                infos.ProcessGradients(sample.Outputs);
                infos.ProcessDeltas(1.0, 0, first);
                first = false;
            }
        }
    }
}