namespace SimpleNN.Core
{
    public interface IGradientLayerInfo : ILayerInfo
    {
        double[] Errors { get; set; }
        double[] Gradients { get; set; }
    }
}