using System.Collections.Generic;

namespace SimpleNN.Core
{
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
                Errors = new double[length];
                Gradients = new double[length];
            }

            public double[] Errors { get; set; }
            public NeuronLayer Layer { get; set; }
            public double[] Gradients { get; set; }
        }

        public RPropLearning(NeuronNet network)
        {
            Network = network;
            // reverse order
            infos = new List<LayerInfo>();
            foreach (var layer in network) infos.Insert(0, new LayerInfo(layer));
        }

        public NeuronNet Network { get; set; }
    }
}