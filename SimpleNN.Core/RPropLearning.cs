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

            bool first = true, notInited = true;

            // collect error
            foreach (var sample in samples)
            {
                // forward step
                Network.Inputs = sample.Inputs;
                Network.Calc();
                // backward
                infos.ProcessGradients(sample.Outputs);
                infos.ProcessDeltas(1.0, 0, first);
                first = false;
                if (notInited)
                {
                    
                }
            }



        }
    }
}