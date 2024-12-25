using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

public class MidasModel : MonoBehaviour
{
    // ���f���ƃe�N�X�`���̊֘A���\�[�X
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
        this.m_Worker = new Worker(this.model, BackendType.GPUCompute);
        this.width = model.inputs[0].shape.Get(3);
        this.height = model.inputs[0].shape.Get(2);
        Debug.Log("input width:  "+this.width+" input_height: "+ this.height);
        Debug.Log(model.inputs[0].shape);
        this.mat=Debugmat;
        
    }

    // 2024/12/26 AI-Tag
    // This was created with assistance from Muse, a Unity Artificial Intelligence product

    private void StandardizedModel(Model model)
    {
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        FunctionalTensor[] outputs = Functional.Forward(model, inputs);

        // Ensure specifying the correct axis for ReduceMax and ReduceMin
        int axis = 0; // Change this to the desired axis for normalization
        FunctionalTensor maxout = Functional.ReduceMax(outputs[0], axis);
        FunctionalTensor minout = Functional.ReduceMin(outputs[0], axis);

        // Normalize the output
        var modelout = (outputs[0] - minout) / (maxout - minout);
        this.model = graph.Compile(modelout);

        
        this.m_Worker = new Worker(this.model, BackendType.GPUCompute);
    }

    private Tensor<float> NormalizeTensor(Tensor<float> input)
    {
        var cpuTensor = input.ReadbackAndClone();
        float max = float.MinValue;
        float min = float.MaxValue;

        for (int i = 0; i < cpuTensor.count; i++)
        {
            float value = cpuTensor[i];
            if (value > max) max = value;
            if (value < min) min = value;
        }
        for (int i = 0;i < cpuTensor.count; i++)
        {
            cpuTensor[i] = (cpuTensor[i] -min)/(max-min) ;
        }
        return cpuTensor;
    }

    public RenderTexture inference(Texture inputTexture)
    {
        RenderTexture outputRendertexture = null; // ������
        try
        {
            Profiler.BeginSample("This is Midas Process");
            
          

            // ���̓e�N�X�`���� Tensor �ɕϊ�
            using (Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture,height: this.height, width: this.width, channels: 3))
            {
                // ���f�����_�̃X�P�W���[�����O
                m_Worker.Schedule(inputTensor);

                // ���f���o�͂��擾
                using (Tensor<float> outputTensor = m_Worker.PeekOutput() as Tensor<float>)
                {
                    if (outputTensor != null)
                    {
                        // �K�v�ɉ����� Tensor �̌`���ύX
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
                        
                        // Tensor �� RenderTexture �ɕϊ�
                        outputRendertexture = TextureConverter.ToTexture(NormalizeTensor(outputTensor));
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