using System;
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
public class MidasEstimation
{
    ModelAsset modelAsset;
    Worker m_Worker;
    Model model;

    public MidasEstimation(ModelAsset modelAsset)
    {
        this.modelAsset = modelAsset;
        this.model = ModelLoader.Load(modelAsset);
        this.m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    public RenderTexture inference(Texture2D inputTexture)
    {
        RenderTexture outputRendertexture = null; // 初期化
        try
        {
            Profiler.BeginSample("This is Midas Process");

            // 入力テクスチャを Tensor に変換
            using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, width: 256, height: 256, channels: 3))
            {
                // モデル推論のスケジューリング
                m_Worker.Schedule(inputTensor);

                // モデル出力を取得
                using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
                {
                    if (outputTensor != null)
                    {
                        // 必要に応じて Tensor の形状を変更
                        outputTensor.Reshape(new TensorShape(1, outputTensor.shape[0], outputTensor.shape[1], outputTensor.shape[2]));
                        Debug.Log($"Output Tensor Shape: {outputTensor.shape}");

                        // Tensor を RenderTexture に変換
                        outputRendertexture = TextureConverter.ToTexture(outputTensor);
                    }
                    else
                    {
                        Debug.LogError("Output tensor is null.");
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during model processing: {e.Message}");
        }
        finally
        {
            Profiler.EndSample();
        }

        return outputRendertexture;

    }

    public void Release()
    {
        m_Worker.Dispose();
    }

}