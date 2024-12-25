using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

public class MidasModel : MonoBehaviour
{
    // モデルとテクスチャの関連リソース
    public ModelAsset modelAsset;
    public RenderTexture inputtex;

    public Material mat;


    MidasEstimation midas;

    void Start()
    {
        midas=new MidasEstimation(modelAsset, mat);
    }

    void Update()
    {
        midas.inference(inputtex);
        
    }

    void OnDisable()
    {
        midas.Release();
    }


}
public class MidasEstimation
{
    ModelAsset modelAsset;
    Worker m_Worker;
    Model model;
    Material mat;

    int width, height;

    public MidasEstimation(ModelAsset modelAsset,Material Debugmat)
    {
        this.modelAsset = modelAsset;
        this.model = ModelLoader.Load(modelAsset);
        this.m_Worker = new Worker(model, BackendType.GPUCompute);
        this.width = model.inputs[0].shape.Get(3);
        this.height = model.inputs[0].shape.Get(2);
        Debug.Log("input width:  "+this.width+" input_height: "+ this.height);
        Debug.Log(model.inputs[0].shape);
        this.mat=Debugmat;
        StandardizedModel(this.model);
    }

    private void  StandardizedModel(Model model)
    {
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        FunctionalTensor[] outputs = Functional.Forward(model, inputs);
        FunctionalTensor maxout= Functional.ReduceMax(outputs[0],0);
        FunctionalTensor minout = Functional.ReduceMin(outputs[0], 0);
        var modelout=(outputs[0] - minout) / (maxout - minout);
        this.model=graph.Compile(modelout);
        Debug.Log(modelout);
        this.m_Worker = new Worker(this.model, BackendType.GPUCompute);
    }

    public RenderTexture inference(Texture inputTexture)
    {
        RenderTexture outputRendertexture = null; // 初期化
        try
        {
            Profiler.BeginSample("This is Midas Process");
            
          

            // 入力テクスチャを Tensor に変換
            using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture,height: this.height, width: this.width, channels: 3))
            {
                // モデル推論のスケジューリング
                m_Worker.Schedule(inputTensor);

                // モデル出力を取得
                using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
                {
                    if (outputTensor != null)
                    {
                        // 必要に応じて Tensor の形状を変更
                        //outputTensor.Reshape(new TensorShape(1, outputTensor.shape[0], outputTensor.shape[1], outputTensor.shape[2]));
                        Debug.Log($"Output Tensor Shape: {outputTensor.shape}");
                        if (outputTensor.shape.rank != 2)
                        {
                            int size = outputTensor.shape.rank;
                            int outwidth= outputTensor.shape[size-1];
                            int outheight= outputTensor.shape[size-2];
                            outputTensor.Reshape(new TensorShape(1,1,outheight, outwidth));
                            Debug.Log($"Changed Output Tensor Shape: {outputTensor.shape}");
                        }
                            
                        // Tensor を RenderTexture に変換
                        outputRendertexture = TextureConverter.ToTexture(outputTensor);
                        this.mat.mainTexture= outputRendertexture;
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