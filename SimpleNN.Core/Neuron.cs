using System;

namespace SimpleNN.Core
{
    public class Neuron
    {
        //protected DoubleFunction Activation;
        public double[] Inputs { get; set; }

        public double[] Weights { get; set; }
        public double Bias { get; set; }

        public double SumValue { get; set; }
        public double OutputValue { get; set; }

        public NeuronLayer Layer { get; set; }

        public Neuron(double[] inputs, NeuronLayer layer)
        {
            Inputs = inputs;
            Layer = layer;
            Weights = new double[inputs.Length];
            Bias = 0.0;
        }

        public double Calc(Action<double> callBack = null)
        {
            var sum = Bias;
            for (int i = 0; i < Weights.Length; i++) sum += Weights[i] * Inputs[i];
            SumValue = sum;
            OutputValue = sum = Layer.Activation(sum);
            if (callBack != null) callBack(sum);
            return sum;
        }
    }
}