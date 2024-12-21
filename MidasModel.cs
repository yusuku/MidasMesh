using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

public class MidasModel : MonoBehaviour
{
    // モデルとテクスチャの関連リソース
    public ModelAsset modelAsset;
    public Texture2D inputtex;
    public RenderTexture outputTexture;
    public Material mat;

    private Worker m_Worker;
    private Model model;

   
        
    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    void Update()
    {
        ProcessModel(inputtex);
    }

    void OnDisable()
    {
        m_Worker.Dispose();
    }

    void ProcessModel(Texture2D inputtex)
    {
        Profiler.BeginSample("This is Midas Process");
        using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputtex, width: 256, height: 256, channels: 3))
        {
            m_Worker.Schedule(inputTensor);
            using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
            {
                if (outputTensor != null)
                {
                    outputTensor.Reshape(new TensorShape(1, outputTensor.shape[0], outputTensor.shape[1], outputTensor.shape[2]));
                    Debug.Log(outputTensor.shape);
                    RenderTexture outputRendertexture= TextureConverter.ToTexture(outputTensor);
                    mat.mainTexture = outputRendertexture;
                }
            }
        }
        Profiler.EndSample();
    }
    
}
