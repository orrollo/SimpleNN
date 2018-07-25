using System;
using System.Collections.Generic;

namespace SimpleNN.Core
{
    public class NeuronLayer : List<Neuron>
    {
        public double[] Inputs { get; set; }
        public double[] Outputs { get; set; }

        public double Activation(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public NeuronLayer PrevLayer { get; set; }
        public NeuronLayer NextLayer { get; set; }

        public bool IsFirstLayer { get { return PrevLayer == null; } }
        public bool IsLastLayer { get { return NextLayer == null; } }

        public NeuronLayer(double[] inputs, int outputCount)
        {
            Inputs = inputs;
            Outputs = new double[outputCount];
            for (int i = 0; i < outputCount; i++) Add(new Neuron(inputs, this));
        }

        public NeuronLayer(NeuronLayer prevLayer, int outputSize)
            : this(prevLayer.Outputs, outputSize)
        {
            PrevLayer = prevLayer;
            prevLayer.NextLayer = this;
        }

        public void Calc()
        {
            for (int i = 0; i < Count; i++) this[i].Calc(val => Outputs[i] = val);
        }

        public double Derivate(double value)
        {
            return value * (1.0 - value);
        }
    }
}