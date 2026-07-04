import React from 'react';
import {AbsoluteFill, Img, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';

const scenes = [
  { id: 'scene_1', imageId: 'demo_2', start: 0.00, duration: 6.00, caption: 'A warm beginning', transition: 'fade', animation: 'slow-zoom' },
  { id: 'scene_2', imageId: 'demo_1', start: 6.00, duration: 6.00, caption: 'Quiet joy', transition: 'fade', animation: 'subtle-pan' },
];

export const Composition: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const active = scenes.find((scene) => frame >= scene.start * fps && frame < (scene.start + scene.duration) * fps) ?? scenes[0];
  return (
    <AbsoluteFill style={{backgroundColor: '#0b0d10', justifyContent: 'center', alignItems: 'center'}}>
      <AbsoluteFill style={{opacity: 0.92}}>
        {active && (
          <Img src={`/assets/${active.imageId}.png`} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
        )}
      </AbsoluteFill>
      <AbsoluteFill style={{justifyContent: 'flex-end', padding: 56, color: 'white'}}>
        <div style={{fontSize: 40, fontWeight: 700, marginBottom: 12}}>{active?.caption}</div>
        <div style={{fontSize: 20, opacity: 0.85}}>fps {fps} · {width}x{height}</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
