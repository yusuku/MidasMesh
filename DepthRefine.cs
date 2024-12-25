using Unity.Sentis;
using UnityEngine;

public class DepthRefine : MonoBehaviour
{
    public Texture2D inputTexture;
    //Midas
    public ModelAsset modelAsset;
    public Material Debugmat;
    MidasEstimation midas;
    RenderTexture outputTexture;

    public Material TrDebugmat;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        midas=new MidasEstimation(modelAsset,Debugmat);
        midas.inference(inputTexture);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    private void OnDestroy()
    {
        midas.Release();
    }
}
