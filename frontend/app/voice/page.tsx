'use client';

import { useState, useRef, useEffect } from 'react';

export default function VoiceMode() {
  const [isRecording, setIsRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [responseText, setResponseText] = useState('');
  const [displayedText, setDisplayedText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    startMicrophone();
    
    return () => {
      cleanup();
    };
  }, []);

  // Typing effect for response text
  useEffect(() => {
    if (!responseText || responseText === displayedText) return;
    
    setDisplayedText(''); // Reset displayed text
    let index = 0;
    
    const timer = setInterval(() => {
      setDisplayedText((prev) => {
        if (index < responseText.length) {
          index++;
          return responseText.slice(0, index);
        } else {
          clearInterval(timer);
          return responseText;
        }
      });
    }, 30); // Typing speed

    return () => clearInterval(timer);
  }, [responseText]);

  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Set up audio visualization
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      visualizeAudio();
      
      console.log('🎤 Microphone ready');
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Please enable microphone access');
    }
  };

  const visualizeAudio = () => {
    if (!analyserRef.current) return;
    
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    const updateLevel = () => {
      analyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setAudioLevel(average);
      animationFrameRef.current = requestAnimationFrame(updateLevel);
    };
    
    updateLevel();
  };

  const startRecording = () => {
    if (!streamRef.current) {
      console.log('⚠️ No microphone stream available');
      return;
    }
    
    if (isRecording || (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording')) {
      console.log('⚠️ Already recording');
      return;
    }
    
    const mediaRecorder = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.start();
    setIsRecording(true);
    console.log('🎤 Recording started');
  };

  const stopRecordingAndSend = async () => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
      console.log('⚠️ No active recording');
      return;
    }

    return new Promise<void>((resolve) => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          
          if (audioBlob.size > 1000) {
            console.log('📤 Sending audio to backend...');
            await sendAudioToBackend(audioBlob);
          } else {
            console.log('⚠️ No audio recorded');
          }
          
          setIsRecording(false);
          resolve();
        };
        
        mediaRecorderRef.current.stop();
      }
    });
  };

  const sendAudioToBackend = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setResponseText('');
    setDisplayedText('');
    
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:8000/process-audio', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.log('⚠️ Backend error:', errorData.error);
        
        // Show error message
        setResponseText('No speech detected. Please try again.');
        setIsProcessing(false);
        return;
      }

      const data = await response.json();
      
      console.log('✅ Got response:', data.text);
      
      // Set response text (typing effect will start automatically)
      setResponseText(data.text);
      
      // Play audio response
      if (data.audio_url) {
        const audio = new Audio(`http://localhost:8000${data.audio_url}`);
        audioRef.current = audio;
        
        audio.onended = () => {
          console.log('✅ Response finished playing');
          setIsProcessing(false);
        };
        
        audio.play();
      } else {
        setIsProcessing(false);
      }
      
    } catch (error) {
      console.error('❌ Error:', error);
      setResponseText('Connection error. Please try again.');
      setIsProcessing(false);
    }
  };

  const toggleMute = async () => {
    if (isProcessing) {
      console.log('⚠️ Still processing, please wait');
      return;
    }
    
    if (isMuted) {
      console.log('▶️ Unmuting - starting recording');
      setIsMuted(false);
      startRecording();
    } else {
      console.log('⏸️ Muting - stopping and sending');
      setIsMuted(true);
      await stopRecordingAndSend();
    }
  };

  const handleInterrupt = () => {
    console.log('⚠️ Interrupting...');
    
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    setIsProcessing(false);
    setIsRecording(false);
    setIsMuted(true);
    setResponseText('');
    setDisplayedText('');
    audioChunksRef.current = [];
    
    console.log('✅ Ready for new question');
  };

  const cleanup = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const handleClose = () => {
    console.log('Close button clicked');
    cleanup();
  };

  // Calculate circle scale based on audio level
  const getCircleScale = () => {
    if (isProcessing) return 1;
    if (!isRecording) return 1;
    const scale = 1 + (audioLevel / 255) * 0.3;
    return Math.min(scale, 1.3);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-emerald-50 via-white to-cyan-50">
      {/* Response Text Display with typing effect */}
      {displayedText && (
        <div className="absolute top-20 max-w-3xl px-8 text-center">
          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-lg p-6 animate-fadeIn">
            <p className="text-xl text-gray-800 leading-relaxed">
              {displayedText}
              {displayedText !== responseText && (
                <span className="inline-block w-1 h-5 bg-teal-500 ml-1 animate-pulse" />
              )}
            </p>
          </div>
        </div>
      )}

      {/* Main Circle with animations */}
      <div className="relative flex items-center justify-center">
        {/* Outer glow rings */}
        {isRecording && (
          <>
            <div className="absolute w-64 h-64 rounded-full bg-green-400/20 animate-ping" />
            <div className="absolute w-72 h-72 rounded-full bg-cyan-400/10 animate-pulse" />
          </>
        )}
        
        {/* Main circle */}
        <div
          className="relative w-48 h-48 rounded-full bg-gradient-to-br from-emerald-500 via-teal-500 to-cyan-500 shadow-2xl transition-all duration-200 flex items-center justify-center"
          style={{
            transform: `scale(${getCircleScale()})`,
          }}
        >
          {/* Inner glow */}
          <div className="absolute inset-4 rounded-full bg-white/20 blur-xl" />
          
          {/* Processing spinner */}
          {isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-32 h-32 border-4 border-white/30 border-t-white rounded-full animate-spin" />
            </div>
          )}
          
          {/* Recording pulse indicator */}
          {isRecording && !isProcessing && (
            <div className="relative z-10">
              <div className="w-16 h-16 rounded-full bg-white animate-pulse" />
            </div>
          )}
          
          {/* Idle state icon */}
          {!isRecording && !isProcessing && (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="white"
              className="w-20 h-20 relative z-10 opacity-80"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
              />
            </svg>
          )}
        </div>
        
        {/* Status text below circle */}
        <div className="absolute -bottom-12 text-center">
          <p className="text-sm font-medium text-gray-700">
            {isProcessing ? (
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" />
                Processing...
                <span className="w-2 h-2 bg-teal-500 rounded-full animate-bounce delay-100" />
              </span>
            ) : isRecording ? (
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Listening...
              </span>
            ) : (
              'Ready'
            )}
          </p>
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-20 flex items-center gap-6">
        {/* Mute/Send Button */}
        <button
          onClick={toggleMute}
          disabled={isProcessing}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl hover:scale-110 ${
            isMuted
              ? 'bg-gradient-to-br from-red-500 to-red-600 hover:from-red-600 hover:to-red-700'
              : 'bg-gradient-to-br from-green-500 to-green-600 hover:from-green-600 hover:to-green-700'
          } ${isProcessing ? 'opacity-50 cursor-not-allowed scale-95' : ''}`}
        >
          {isMuted ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2.5}
              stroke="white"
              className="w-8 h-8"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 19L5 5m14 0L5 19" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2.5}
              stroke="white"
              className="w-8 h-8"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
              />
            </svg>
          )}
        </button>

        {/* Interrupt Button */}
        <button
          onClick={handleInterrupt}
          className="w-20 h-20 rounded-full bg-gradient-to-br from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 flex items-center justify-center transition-all duration-300 shadow-xl hover:scale-110"
          title="Interrupt and start over"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
            stroke="white"
            className="w-8 h-8"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"
            />
          </svg>
        </button>

        {/* Close Button */}
        <button
          onClick={handleClose}
          className="w-20 h-20 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 hover:from-gray-800 hover:to-gray-900 flex items-center justify-center transition-all duration-300 shadow-xl hover:scale-110"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
            stroke="white"
            className="w-8 h-8"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-6 text-center">
        <p className="text-sm text-gray-600 font-medium">
          {isMuted ? '🔴 Click to start recording' : '🟢 Click to stop & send'} • 🟠 Interrupt
        </p>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}