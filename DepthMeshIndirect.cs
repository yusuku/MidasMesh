using System;
using System.Runtime.InteropServices;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngineInternal;


public class DepthMeshIndirect : MonoBehaviour
{
    public Texture2D inputTexture;

    //Depth Estimation
    public ModelAsset modelAsset;

    public Material material;
    public RenderTexture Depthmap;

    //GPUInstancing

    public Material InstanceMaterial;
    public Mesh InstanceMesh;
    public ComputeShader InstanceCS;

    MidasEstimation midas;
    TextureDepthGPUInstancing textureDepthGPUInstancing;
    void Start()
    {
        midas = new MidasEstimation(modelAsset);
        Depthmap = midas.inference(inputTexture);
        material.mainTexture = Depthmap;
        textureDepthGPUInstancing = new TextureDepthGPUInstancing(InstanceMaterial,InstanceMesh,InstanceCS, inputTexture);
    }

    void Update()
    {
        
        textureDepthGPUInstancing.DrawMeshes(Depthmap);
    }
    void OnDestroy()
    {
        midas.Release();
        textureDepthGPUInstancing.Release();
    }



}


public class TextureDepthGPUInstancing
{
    private readonly Material material; // �`��Ɏg�p����}�e���A��
    private readonly Mesh mesh; // �`��Ɏg�p���郁�b�V��
    private readonly ComputeShader computeShader;

    private readonly Texture2D diffuseMap; // ���͊g�U�e�N�X�`��

    private GraphicsBuffer commandBuf;
    private GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;

    private GraphicsBuffer positionBuffer;
    private GraphicsBuffer colorBuffer;

    private readonly int kernelId;
    private readonly int width;
    private readonly int height;
    private readonly uint xThread;
    private readonly uint yThread;

    public TextureDepthGPUInstancing(Material material, Mesh mesh, ComputeShader computeShader, Texture2D diffuseMap)
    {
        this.material = material ;
        this.mesh = mesh;
        this.computeShader = computeShader;
        this.diffuseMap = diffuseMap;


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
        computeShader.SetBuffer(kernelId, "PositionResult", positionBuffer);
        computeShader.SetBuffer(kernelId, "ColorResult", colorBuffer);
    }

    public void DrawMeshes(RenderTexture depthmap)
    {
        try
        {
            computeShader.SetTexture(kernelId, "DepthTexture", depthmap);
            // Compute Shader �̃f�B�X�p�b�`
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