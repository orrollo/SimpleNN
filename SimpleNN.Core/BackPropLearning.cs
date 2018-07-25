using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleNN.Core
{
    public class BackPropLearning
    {
        private List<LayerInfo> infos;

        public BackPropLearning(NeuronNet network)
        {
            Network = network;
            // reverse order
            infos = new List<LayerInfo>();
            foreach (var layer in network) infos.Insert(0, new LayerInfo(layer));
        }

        public NeuronNet Network { get; set; }

        class LayerInfo : IGradientLayerInfo
        {
            private object _gradients;

            public LayerInfo(NeuronLayer layer)
            {
                Layer = layer;
                var length = layer.Outputs.Length;
                Errors = new double[length];
                Gradients = new double[length];
                PrevDeltas = new double[length, layer.Inputs.Length];
                PrevBiasDeltas = new double[length];
            }

            public double[] Errors { get; set; }

            public NeuronLayer Layer { get; set; }

            public double[] Gradients { get; set; }

            public double[,] PrevDeltas { get; set; }

            public double[] PrevBiasDeltas { get; set; }
        }

        public void TrainNetwork(DataSamples samples, DataSamples testSamples, BackPropParams param)
        {
            var rnd = new Random();
            //
            var items = new List<int>();
            for (int i = 0; i < samples.Count; i++) items.Add(i);
            //
            InitNeurons(rnd);
            var stop = false;
            var step = 0;
            double error = 0.0;
            while (!stop)
            {
                for (int i = 0; i < items.Count; i++)
                {
                    int j = rnd.Next(items.Count);
                    while (i == j) j = rnd.Next(items.Count);
                    int tmp = items[i];
                    items[i] = items[j];
                    items[j] = tmp;
                }
                for (int i = 0; i < items.Count; i++) TrainBySample(samples[items[i]], param.Eta, param.Alpha);
                //
                error = Error(testSamples);
                step++;
                if (param.CallBack != null) param.CallBack(step, error, false);
                stop = (step >= param.MaxSteps) || (error <= param.ErrorStopValue);
            }
            if (param.CallBack != null) param.CallBack(step, error, true);
        }

        private void InitNeurons(Random rnd)
        {
            foreach (var info in infos)
            {
                var layer = info.Layer;
                foreach (var neuron in layer)
                {
                    neuron.Bias = rnd.NextDouble() - 0.5;
                    for (int i = 0; i < neuron.Weights.Length; i++) neuron.Weights[i] = rnd.NextDouble() - 0.5;
                }
            }
            if (Network.Count < 2) return;

            int n = Network[0].Inputs.Length, p = 0;
            for (int index = 0; index < (Network.Count - 1); index++) p += Network[index].Outputs.Length;
            double beta = 0.7*Math.Pow(p, 1.0/n);

            foreach (var info in infos)
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

        protected void TrainBySample(DataSample data, double eta, double alpha)
        {
            // forward step
            Network.Inputs = data.Inputs;
            Network.Calc();
            // backward
            infos.ProcessGradients(data.Outputs);
            UpdateWeights(eta, alpha);
        }

        public double Error(DataSamples testSamples)
        {
            double sum = 0.0;
            for (int index = 0; index < testSamples.Count; index++)
            {
                double[] input = testSamples[index].Inputs, output = testSamples[index].Outputs;
                Network.Inputs = input;
                Network.Calc();
                var error = 0.0;
                for (int itemIntex = 0; itemIntex < output.Length; itemIntex++)
                {
                    var diff = output[itemIntex] - Network.Outputs[itemIntex];
                    error += diff*diff;
                }
                error = Math.Sqrt(error)/output.Length;
                sum += error;
            }
            return sum/testSamples.Count;
        }

        private void UpdateWeights(double eta, double alpha)
        {
            for (int layerIndex = 0; layerIndex < infos.Count; layerIndex++)
            {
                var info = infos[layerIndex];
                var layer = info.Layer;
                for (int neuronIndex = 0; neuronIndex < layer.Count; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];
                    var neuronKoef1 = eta*info.Gradients[neuronIndex]*(1.0-alpha);
                    for (int weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        double delta = neuronKoef1*layer.Inputs[weightIndex];
                        neuron.Weights[weightIndex] += delta;
                        neuron.Weights[weightIndex] += alpha * info.PrevDeltas[neuronIndex, weightIndex];
                        info.PrevDeltas[neuronIndex, weightIndex] = delta;
                    }
                    var biasDelta = neuronKoef1*1.0;
                    neuron.Bias += biasDelta;
                    neuron.Bias += alpha*info.PrevBiasDeltas[neuronIndex];
                    info.PrevBiasDeltas[neuronIndex] = biasDelta;
                }
            }
        }

        //private void ProcessGradients(double[] outputs)
        //{
        //    for (int i = 0; i < infos.Count; i++)
        //    {
        //        var info = infos[i];
        //        var layer = info.Layer;
        //        var values = layer.Outputs;
        //        if (i == 0)
        //        {
        //            // output layer
        //            for (int j = 0; j < values.Length; j++)
        //            {
        //                var value = values[j];
        //                info.Errors[j] = outputs[j] - value;
        //                info.Gradients[j] = layer.Derivate(value)*info.Errors[j];
        //            }
        //        }
        //        else
        //        {
        //            // hidden layers
        //            var prev = infos[i - 1];
        //            for (int j = 0; j < values.Length; j++)
        //            {
        //                var value = values[j];
        //                double sum = 0.0;
        //                for (int k = 0; k < prev.Layer.Count; k++) sum += prev.Layer[k].Weights[j]*prev.Gradients[k];
        //                info.Errors[j] = sum;
        //                info.Gradients[j] = layer.Derivate(value)*info.Errors[j];
        //            }
        //        }
        //    }
        //}
    }
}