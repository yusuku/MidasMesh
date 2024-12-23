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


   
        
    void Start()
    {
       
    }

    void Update()
    {

    }

    void OnDisable()
    {
  
    }

    
}
public class MidasEstimation
{
    ModelAsset modelAsset;
    Worker m_Worker;
    Model model;
    Material Debugmat;

    public MidasEstimation(ModelAsset modelAsset,Material Debugmat)
    {
        this.Debugmat = Debugmat;
        this.modelAsset = modelAsset;
        this.model = ModelLoader.Load(modelAsset);
        this.m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    public RenderTexture inference(RenderTexture inputTexture)
    {
        RenderTexture outputRendertexture = null; // ������
        try
        {
            Profiler.BeginSample("This is Midas Process");

            // ���̓e�N�X�`���� Tensor �ɕϊ�
            using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, width: 1024, height: 512, channels: 3))
            {
                // ���f�����_�̃X�P�W���[�����O
                m_Worker.Schedule(inputTensor);

                // ���f���o�͂��擾
                using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
                {
                    if (outputTensor != null)
                    {
                        // �K�v�ɉ����� Tensor �̌`���ύX
                        
                        Debug.Log($"Output Tensor Shape: {outputTensor.shape}");

                        // Tensor �� RenderTexture �ɕϊ�
                        outputRendertexture = TextureConverter.ToTexture(outputTensor);
                        this.Debugmat.mainTexture = outputRendertexture;
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