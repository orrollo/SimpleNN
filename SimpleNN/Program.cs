using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimpleNN.Core;

namespace SimpleNN
{
    class Program
    {
        private static readonly DataSamples XorSamples = new DataSamples()
        {
            new DataSample(2, 0.0, 0.0, 0.0),
            new DataSample(2, 0.0, 1.0, 1.0),
            new DataSample(2, 1.0, 0.0, 1.0),
            new DataSample(2, 1.0, 1.0, 0.0)
        };

        private static NeuronNet _network;

        static void Main(string[] args)
        {
            _network = new NeuronNet(2, 3, 1);
            var learning = new BackPropLearning(_network);
            var param = new BackPropParams()
            {
                CallBack = cb,
                Eta = 0.9,
                Alpha = 0.05,
                ErrorStopValue = 0.05
            };
            learning.TrainNetworkBySample(XorSamples, XorSamples, param);
            //learning.TrainNetworkByBatch(XorSamples, XorSamples, param);

            //var param2 = new RPropParams();
            //var learning2 = new RPropLearning(_network);
            //learning2.Train(XorSamples, XorSamples, param2);

            Console.WriteLine("press enter...");
            Console.ReadLine();
        }

        private static void cb(int step, double currenterror, bool isfinal)
        {
            if (step > 1 && !isfinal && (step % 10) != 0) return;
            Console.WriteLine("step {0}, error = {1:F4}, final = {2}", step, currenterror, isfinal ? "true" : "false");
            if (step > 1 && !isfinal && (step % 100) != 0) return;
            foreach (var sample in XorSamples)
            {
                _network.Inputs = sample.Inputs;
                _network.Calc();
                var inps = string.Join(";", sample.Inputs.Select(x => x.ToString("f2")));
                var value = _network.Outputs[0];
                Console.WriteLine("{0} => {1:f2} error {2:f3}", inps, value, sample.Outputs[0] - value);
            }
        }
    }
}
