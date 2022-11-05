import type { NextPage } from "next";
import Head from "next/head";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs"
import { ImageDataToTensor } from "../utils";

const Home: NextPage = () => {

  const videoRef = useRef<HTMLVideoElement>(null);
  const photoRef = useRef<HTMLCanvasElement>(null);
  const [photos, setPhotos] = useState<[string | null, string | null]>([null,null])
  const [model, setModel] = useState<tf.GraphModel<string | tf.io.IOHandler>>();
  const [tensorData, setTensorData] = useState<{start: null | tf.Tensor3D, end:null | tf.Tensor3D}>({
    start: null,
    end: null
  })
  const getVideo = ()=>{
    navigator.mediaDevices.getUserMedia({
      video: {width: 720, height: 405}
    }).then(stream=>{
      let video = videoRef.current; 
      video!.srcObject = stream;
      video!.play()
    }).catch(err=>{
      console.error(err)
    })
  }
  useEffect(()=>{
    getVideo();
  }, [videoRef])

  useEffect(()=>{
    (async()=>{let model = await tf.loadGraphModel("film_net/Style/saved_model/model.json");
    setModel(model);
    console.log(model)})();
  }, [])

  const takePhoto = async()=>{
    let photo = photoRef.current;
    const width = 1920;
    const height = width / (16/9);


    photo!.width = width;
    photo!.height = height;
    let ctx = photoRef.current!.getContext("2d");
    ctx!.drawImage(videoRef.current!, 0, 0, width, height);
    let img = photoRef.current!.toDataURL();

    let img_tensor = await ImageDataToTensor(ctx!.getImageData(0,0,width,height))
    setTensorData(prev=>{
      if (!prev["start"]){
        return {
          ...prev,
          ["start"]: img_tensor
        }
      } else {
        return {
          ...prev,
          ["end"]: img_tensor
        }
      }
    })

    // let blob = new Blob([img.data as Uint8Array], {"type": "image/png"})
    // let url = URL.createObjectURL(blob);

    setPhotos(prev=>{
      let mew: [string| null, string| null] = [...prev]
      if (prev[0] != null){

        mew[1] = img;
      } else {
        mew[0] = img;
      }
      return mew;
    });
    
    


  }

  useEffect(()=>{
    console.log(photos)
  }, [photos])

  const interp = () =>{

    let out = model!.execute([tensorData.start!, tensorData.end!, tf.tensor1d([1])], )
    console.log(out)


  }

  return (
    <>
    <div className="w-full h-screen bg-black flex flex-col items-center">
      <h1 className="mt-8 mb-12 bg-gradient-to-r font-light text-5xl from-purple-500 to-cyan-500 bg-clip-text text-transparent ">fuse_cam v0.0.1</h1>
      <video ref={videoRef}></video>
      <canvas className="hidden" ref={photoRef}/>
      <div className="w-full flex flex-row justify-center">
        {photos[0] && 

          <Image className="relative" src={photos[0]} width={480} height={270}></Image>

        }
        {photos[1] && 
          <Image className="relative" src={photos[1]} width={480} height={270}></Image>
        }
        {/* {photos[1] && <Image src={photos[1] && photos[1]}></Image>} */}
      </div>
      {!(photos[0] && photos[1]) ? (<button  className="bg-purple-500 w-24 h-16 rounded-md"onClick={takePhoto}>Click</button>) : 
      (
        <button 
          className="w-32 h-16 rounded-md bg-gradient-to-r from-purple-500 to-orange-500 text-white font-light"
          onClick={()=>interp()}
        >Interpolate</button>
      )}
    </div>
    </>
  );
};

export default Home;

type TechnologyCardProps = {
  name: string;
  description: string;
  documentation: string;
};

const TechnologyCard = ({
  name,
  description,
  documentation,
}: TechnologyCardProps) => {
  return (
    <section className="flex flex-col justify-center rounded border-2 border-gray-500 p-6 shadow-xl duration-500 motion-safe:hover:scale-105">
      <h2 className="text-lg text-gray-700">{name}</h2>
      <p className="text-sm text-gray-600">{description}</p>
      <a
        className="m-auto mt-3 w-fit text-sm text-violet-500 underline decoration-dotted underline-offset-2"
        href={documentation}
        target="_blank"
        rel="noreferrer"
      >
        Documentation
      </a>
    </section>
  );
};
