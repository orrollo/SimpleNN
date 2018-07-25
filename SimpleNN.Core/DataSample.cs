namespace SimpleNN.Core
{
    public class DataSample
    {
        public double[] Inputs { get; set; }
        public double[] Outputs { get; set; }

        public DataSample(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public DataSample(int inputsize, params double[] values)
        {
            Inputs = new double[inputsize];
            Outputs = new double[values.Length - inputsize];
            for (int i = 0; i < values.Length; i++)
            {
                if (i < inputsize)
                    Inputs[i] = values[i];
                else
                    Outputs[i - inputsize] = values[i];
            }
        }
    }
}