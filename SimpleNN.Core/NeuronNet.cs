using System;
using System.Collections.Generic;

namespace SimpleNN.Core
{
    public class NeuronNet : List<NeuronLayer>
    {
        private double[] _inputs;

        public double[] Inputs
        {
            get { return _inputs; }
            set
            {
                if (value == null || _inputs.Length != value.Length) throw new ArgumentException();
                if (value != _inputs) for (int i = 0; i < value.Length; i++) _inputs[i] = value[i];
            }
        }

        public double[] Outputs { get; set; }

        public NeuronNet(params int[] sizes)
        {
            if (sizes == null || sizes.Length < 2) throw new ArgumentException();
            _inputs = new double[sizes[0]];
            for (int i = 1; i < sizes.Length; i++)
            {
                var layer = i == 1 ? new NeuronLayer(Inputs, sizes[i]) : new NeuronLayer(this[i - 2], sizes[i]);
                Outputs = layer.Outputs;
                Add(layer);
            }
        }

        public void Calc()
        {
            foreach (var layer in this) layer.Calc();
        }

        public double Error(DataSamples testSamples)
        {
            double sum = 0.0;
            for (int index = 0; index < testSamples.Count; index++)
            {
                double[] input = testSamples[index].Inputs, output = testSamples[index].Outputs;
                Inputs = input;
                Calc();
                var error = 0.0;
                for (int itemIntex = 0; itemIntex < output.Length; itemIntex++)
                {
                    var diff = output[itemIntex] - Outputs[itemIntex];
                    error += diff*diff;
                }
                error = Math.Sqrt(error)/output.Length;
                sum += error;
            }
            return sum/testSamples.Count;
        }
    }
}