using System;
using System.Runtime.InteropServices;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngineInternal;


public class DepthMeshIndirect : MonoBehaviour
{
    public RenderTexture inputTexture;

    //Midas
    public ModelAsset modelAsset;
    //Meshinstance
    public Material Instancematerial;
    public Mesh Instancemesh;
    public ComputeShader InstanceShader;

    public Material Debugmat;

    MidasEstimation midas;
    RenderTexture outputTexture;
    TextureDepthGPUInstancing GPUInstancing;
    //GPUInstancing

    void Start()
    {
        midas = new MidasEstimation(modelAsset,Debugmat);
        outputTexture= midas.inference(inputTexture);
        
        GPUInstancing = new TextureDepthGPUInstancing(Instancematerial, Instancemesh, InstanceShader, inputTexture, outputTexture);

    }

    void Update()
    {
        outputTexture = midas.inference(inputTexture);

        GPUInstancing.DrawMeshes(outputTexture);
    }
    void OnDestroy()
    {
        GPUInstancing.Release();
        midas.Release();
    }
}

public class TextureDepthGPUInstancing
{
    private readonly Material material; // �`��Ɏg�p����}�e���A��
    private readonly Mesh mesh; // �`��Ɏg�p���郁�b�V��
    private readonly ComputeShader computeShader;

    private readonly RenderTexture diffuseMap; // ���͊g�U�e�N�X�`��
    private RenderTexture Depthmap;

    private GraphicsBuffer commandBuf;
    private GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;

    private GraphicsBuffer positionBuffer;
    private GraphicsBuffer colorBuffer;

    private readonly int kernelId;
    private readonly int width;
    private readonly int height;
    private readonly uint xThread;
    private readonly uint yThread;

    public TextureDepthGPUInstancing(Material material, Mesh mesh, ComputeShader computeShader, RenderTexture diffuseMap,RenderTexture Depthmap)
    {
        this.material = material ;
        this.mesh = mesh;
        this.computeShader = computeShader;
        this.diffuseMap = diffuseMap;
        this.Depthmap = Depthmap;

        this.width = diffuseMap.width;
        this.height = diffuseMap.height;

        // �R�}���h�o�b�t�@�̏�����
        commandBuf = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[1]
        {
            new GraphicsBuffer.IndirectDrawIndexedArgs
            {
                indexCountPerInstance = mesh.GetIndexCount(0),
                instanceCount = (uint)(width * height),
            }
        };
        commandBuf.SetData(commandData);

        // �o�b�t�@�̏�����
        positionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, width * height, Marshal.SizeOf<Vector3>());
        colorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, width * height, Marshal.SizeOf<Color>());

        // Compute Shader �̃Z�b�g�A�b�v
        kernelId = computeShader.FindKernel("CSMain");
        computeShader.SetInt("width", width);
        computeShader.SetInt("height", height);
        computeShader.GetKernelThreadGroupSizes(kernelId, out xThread, out yThread, out _);
        computeShader.SetTexture(kernelId, "DiffuseTexture", diffuseMap);
        computeShader.SetTexture(kernelId, "DepthTexture", Depthmap);
        computeShader.SetBuffer(kernelId, "PositionResult", positionBuffer);
        computeShader.SetBuffer(kernelId, "ColorResult", colorBuffer);
        computeShader.Dispatch(kernelId, Mathf.CeilToInt(width / (float)xThread), Mathf.CeilToInt(height / (float)yThread), 1);

    }

    public void DrawMeshes(RenderTexture depthmap)
    {
        try
        {
            this.Depthmap = depthmap;
            computeShader.SetTexture(kernelId, "DepthTexture", Depthmap);
            computeShader.Dispatch(kernelId, Mathf.CeilToInt(width / (float)xThread), Mathf.CeilToInt(height / (float)yThread), 1);
            // �`��̃p�����[�^��ݒ�
            RenderParams renderParams = new RenderParams(material)
            {
                worldBounds = new Bounds(Vector3.zero, 10000 * Vector3.one), // �傫�ȃo�E���h��ݒ�
                matProps = new MaterialPropertyBlock()
            };

            renderParams.matProps.SetBuffer("_Positions", positionBuffer);
            renderParams.matProps.SetBuffer("_Colors", colorBuffer);

            // �Ԑڕ`��
            Graphics.RenderMeshIndirect(renderParams, mesh, commandBuf);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during DrawMeshes: {e.Message}");
        }
    }

    public void Release()
    {
        commandBuf?.Release();
        commandBuf = null;

        positionBuffer?.Release();
        positionBuffer = null;

        colorBuffer?.Release();
        colorBuffer = null;

        Debug.Log("Resources released successfully.");
    }
}