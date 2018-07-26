namespace SimpleNN.Core
{
    public interface IGradientLayerInfo : ILayerInfo
    {
        double[] Gradients { get; set; }
        double[,] Deltas { get; set; }
        double[] BiasDeltas { get; set; }
        double[,] PrevDeltas { get; }
        double[] PrevBiasDeltas { get; }
    }
}