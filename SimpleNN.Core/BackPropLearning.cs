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
                Gradients = new double[length];
                //
                PrevDeltas = new double[length, layer.Inputs.Length];
                PrevBiasDeltas = new double[length];
                //
                Deltas = new double[length, layer.Inputs.Length];
                BiasDeltas = new double[length];
            }

            public NeuronLayer Layer { get; set; }

            public double[] Gradients { get; set; }

            public double[,] Deltas { get; set; }
            public double[] BiasDeltas { get; set; }

            public double[,] PrevDeltas { get; private set; }
            public double[] PrevBiasDeltas { get; private set; }
        }

        public void TrainNetworkByBatch(DataSamples samples, DataSamples testSamples, BackPropParams param)
        {
            var rnd = new Random();
            //
            infos.InitNeurons(rnd);
            infos.InitNG(Network, rnd);
            var stop = false;
            var step = 0;
            double error = 0.0;
            var count = samples.Count;
            while (!stop)
            {
                for (int sampleIndex = 0; sampleIndex < count; sampleIndex++)
                {
                    var data = samples[sampleIndex];
                    // forward step
                    Network.Inputs = data.Inputs;
                    Network.Calc();
                    // backward
                    infos.ProcessGradients(data.Outputs/*, sampleIndex == 0*/);
                    infos.ProcessDeltas(param.Eta, param.Alpha, sampleIndex == 0);
                }
                infos.UpdateWeights(param.Alpha);
                error = Network.Error(testSamples);
                step++;
                if (param.CallBack != null) param.CallBack(step, error, false);
                stop = (step >= param.MaxSteps) || (error <= param.ErrorStopValue);
            }
            if (param.CallBack != null) param.CallBack(step, error, true);
        }

        public void TrainNetworkBySample(DataSamples samples, DataSamples testSamples, BackPropParams param)
        {
            var rnd = new Random();
            var items = GenerateIndexes(samples);
            //
            infos.InitNeurons(rnd);
            infos.InitNG(Network, rnd);
            var stop = false;
            var step = 0;
            double error = 0.0;
            while (!stop)
            {
                ShuffleIndexes(items, rnd);
                for (int i = 0; i < items.Count; i++) TrainBySample(samples[items[i]], param.Eta, param.Alpha);
                //
                error = Network.Error(testSamples);
                step++;
                if (param.CallBack != null) param.CallBack(step, error, false);
                stop = (step >= param.MaxSteps) || (error <= param.ErrorStopValue);
            }
            if (param.CallBack != null) param.CallBack(step, error, true);
        }

        private static List<int> GenerateIndexes(DataSamples samples)
        {
            var items = new List<int>();
            for (int i = 0; i < samples.Count; i++) items.Add(i);
            return items;
        }

        private static void ShuffleIndexes(List<int> items, Random rnd)
        {
            for (int i = 0; i < items.Count; i++)
            {
                int j = rnd.Next(items.Count);
                while (i == j) j = rnd.Next(items.Count);
                int tmp = items[i];
                items[i] = items[j];
                items[j] = tmp;
            }
        }

        protected void TrainBySample(DataSample data, double eta, double alpha)
        {
            // forward step
            Network.Inputs = data.Inputs;
            Network.Calc();
            // backward
            infos.ProcessGradients(data.Outputs);
            infos.ProcessDeltas(eta, alpha);
            infos.UpdateWeights(alpha);
        }
    }
}