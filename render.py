"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ffmpeg
import numpy as np
import torch as th
import cv2
from pytorch3d.io import load_obj, load_ply, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
)


class Renderer:
    def __init__(self, faces, n_vertices):
        """
        :param faces: long tensor with n*3
        :param n_vertices: int
        """
        if th.cuda.is_available():
            self.device = th.device("cuda:0")
            th.cuda.set_device(self.device)
        else:
            self.device = th.device("cpu")
    
        #print(f"verts.shape: {verts.shape}")
        #print(f"verts.max(): {verts.max()}, verts.min(): {verts.min()}")
        #print(f"verts.mean(): {verts.mean()}, verts.std(): {verts.std()}")
        # self.verts_rgb = (th.ones((n_vertices, 3)) * th.Tensor([0.529, 0.807, 0.980]).unsqueeze(0)).unsqueeze(0)
        self.verts_rgb = (th.ones((n_vertices, 3)) * th.Tensor([1, 1, 1]).unsqueeze(0)).unsqueeze(0)
        self.verts_rgb = self.verts_rgb.to(self.device)
        self.faces = faces.to(self.device).unsqueeze(0)

    def render(self, verts: th.Tensor):
        """
        :param verts: B x V x 3 tensor containing a batch of face vertex positions to be rendered
        :return: B x 640 x 480 x 4 tensor containing the rendered images
        """
        R, T = look_at_view_transform(7.5, 0, 20)
        focal = th.tensor([5.0], dtype=th.float32).to(self.device)
        princpt = th.tensor([0.1, 0.1], dtype=th.float32).to(self.device).unsqueeze(0)

        cameras = PerspectiveCameras(device=self.device, focal_length=focal, R=R, T=T, principal_point=princpt)

        raster_settings = RasterizationSettings(
            image_size=[640, 480],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[0.0, 0.0, 10.0]])

        verts = verts * 10
        textures = Textures(verts_rgb=self.verts_rgb.expand(verts.shape[0], -1, -1))
        #print(f"verts.shape: {verts.shape}")
        #print(f"faces.shape: {self.faces.shape}")
        #print(f"after, faces.shape: {self.faces.expand(verts.shape[0], -1, -1).shape}")
        mesh = Meshes(
            verts=verts.to(self.device),
            faces=self.faces.expand(verts.shape[0], -1, -1),
            textures=textures
        )

        with th.no_grad():
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=lights
                )
            )
            images = renderer(mesh)

        return images

    def to_meshes(self, verts: th.Tensor, video_output: str):
        print(f"faces.shape: {self.faces.shape}")
        for i in range(verts.shape[0]):
            save_ply(video_output+'_'+str(i)+'.ply', verts[i], self.faces.squeeze(0))

    def to_image(self, verts: th.Tensor, image_output: str):
        images = self.render(verts)
        images = 255 * images[:, :, :, :3].cpu().contiguous().numpy()
        images = images.astype(np.uint8)
        cv2.imwrite(image_output, images[0])
        # np.save(image_output, images)

    def to_video(self, verts: th.Tensor, audio_file: str, video_output: str, fps: int = 30, batch_size: int = 30):
        """
        :param verts: B x V x 3 tensor containing a batch of face vertex positions to be rendered
        :param audio_file: filename of the audio input file
        :param video_output: filename of the output video file
        :param fps: frame rate of output video
        :param batch_size: number of frames to render simultaneously in one batch
        """
        if not video_output[-4:] == '.mp4':
            video_output = video_output + '.mp4'

        images = th.cat([self.render(v).cpu() for v in th.split(verts, batch_size)], dim=0)
        images = 255 * images[:, :, :, :3].contiguous().numpy()
        images = images.astype(np.uint8)
        #for i in range(images.shape[0]):
        #    Image.fromarray(images[i]).save(f'{video_output}_{i}.png')
        #print(f"images.shape: {images.shape}")

        video_stream = ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="480x640", r=fps)
        audio_stream = ffmpeg.input(filename=audio_file)
        streams = [video_stream, audio_stream]
        output_args = {
            "format": "mp4",
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "movflags": "frag_keyframe+empty_moov+faststart"
        }
        proc = (
            ffmpeg
            .output(*streams, video_output, **output_args)
            .overwrite_output()
            .global_args("-loglevel", "fatal")
            .run_async(pipe_stdin=True, pipe_stdout=False)
        )

        proc.communicate(input=images.tobytes())
