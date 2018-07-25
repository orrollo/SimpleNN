namespace SimpleNN.Core
{
    public class BackPropParams
    {
        private double _eta = 0.9;
        private double _alpha = 0.05;
        private int _maxSteps = int.MaxValue;
        private double _errorStopValue = 0.1;

        public delegate void TrainCallback(int step, double currentError, bool isFinal);

        public double Eta
        {
            get { return _eta; }
            set { _eta = value; }
        }

        public double Alpha
        {
            get { return _alpha; }
            set { _alpha = value; }
        }

        public TrainCallback CallBack { get; set; }

        public int MaxSteps
        {
            get { return _maxSteps; }
            set { _maxSteps = value; }
        }

        public double ErrorStopValue
        {
            get { return _errorStopValue; }
            set { _errorStopValue = value; }
        }
    }
}