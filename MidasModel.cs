using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

public class MidasModel : MonoBehaviour
{
    // ���f���ƃe�N�X�`���̊֘A���\�[�X
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
        RenderTexture outputRendertexture = null; // ������
        try
        {
            Profiler.BeginSample("This is Midas Process");

            // ���̓e�N�X�`���� Tensor �ɕϊ�
            using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, width: 256, height: 256, channels: 3))
            {
                // ���f�����_�̃X�P�W���[�����O
                m_Worker.Schedule(inputTensor);

                // ���f���o�͂��擾
                using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
                {
                    if (outputTensor != null)
                    {
                        // �K�v�ɉ����� Tensor �̌`���ύX
                        outputTensor.Reshape(new TensorShape(1, outputTensor.shape[0], outputTensor.shape[1], outputTensor.shape[2]));
                        Debug.Log($"Output Tensor Shape: {outputTensor.shape}");

                        // Tensor �� RenderTexture �ɕϊ�
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