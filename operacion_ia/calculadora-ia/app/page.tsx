"use client";
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Calculadora() {
  // CORRECCIÓN: Agregamos <any> para evitar el error de TypeScript "Argument of type LayersModel..."
  const [modeloActual, setModeloActual] = useState<any>(null);
  const [operacion, setOperacion] = useState('suma'); // 'suma' o 'resta'
  const [valA, setValA] = useState('');
  const [valB, setValB] = useState('');
  const [resultado, setResultado] = useState(null);
  const [estado, setEstado] = useState('Iniciando...');

  // Cargar el modelo cuando cambia la operación
  useEffect(() => {
    async function cambiarModelo() {
      setModeloActual(null); 
      setResultado(null);
      setEstado(`Cargando cerebro de ${operacion}...`);
      
      const path = operacion === 'suma' 
        ? '/modelo_suma/model.json' 
        : '/modelo_resta/model.json';
      
      try {
        const m = await tf.loadLayersModel(path);
        setModeloActual(m);
        setEstado(`Modo: ${operacion.toUpperCase()} listo 🚀`);
      } catch (err) {
        console.error(err);
        setEstado('Error al cargar el modelo');
      }
    }
    cambiarModelo();
  }, [operacion]); 

  const calcular = async () => {
    if (!modeloActual || valA === '' || valB === '') return;

    try {
        // TENSORFLOW.JS: Crear tensor de entrada [1, 2] (Batch size 1, 2 features)
        const input = tf.tensor2d([[parseFloat(valA), parseFloat(valB)]]);
        
        const prediccion = modeloActual.predict(input);
        const data = await prediccion.data();
        
        setResultado(data[0].toFixed(2));
        
        // Limpieza de memoria
        input.dispose();
        prediccion.dispose();
    } catch (e) {
        console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl w-full max-w-md border border-gray-700">
        <h1 className="text-3xl font-bold mb-2 text-center text-purple-400">Calculadora Neuronal</h1>
        <p className="text-gray-400 text-center mb-6 text-sm">Sin operadores matemáticos (+ o -)</p>

        {/* Estado */}
        <div className={`text-center mb-6 p-2 rounded font-mono text-sm ${modeloActual ? 'bg-green-900 text-green-200' : 'bg-yellow-900 text-yellow-200'}`}>
            {estado}
        </div>

        {/* Selector */}
        <div className="flex gap-4 mb-6">
          <button onClick={() => setOperacion('suma')}
            className={`flex-1 py-3 rounded font-bold transition-all ${operacion === 'suma' ? 'bg-blue-600 ring-2 ring-blue-400' : 'bg-gray-700 hover:bg-gray-600'}`}>
            Suma (+)
          </button>
          <button onClick={() => setOperacion('resta')}
            className={`flex-1 py-3 rounded font-bold transition-all ${operacion === 'resta' ? 'bg-red-600 ring-2 ring-red-400' : 'bg-gray-700 hover:bg-gray-600'}`}>
            Resta (-)
          </button>
        </div>

        {/* Inputs */}
        <div className="space-y-4">
          <div className="flex gap-4">
            <input type="number" value={valA} onChange={(e) => setValA(e.target.value)}
              placeholder="A" className="w-1/2 p-3 rounded bg-gray-700 border border-gray-600 focus:ring-2 focus:ring-purple-500 outline-none text-center text-xl"/>
            <input type="number" value={valB} onChange={(e) => setValB(e.target.value)}
              placeholder="B" className="w-1/2 p-3 rounded bg-gray-700 border border-gray-600 focus:ring-2 focus:ring-purple-500 outline-none text-center text-xl"/>
          </div>
          
          <button onClick={calcular} disabled={!modeloActual}
            className="w-full py-4 bg-purple-600 hover:bg-purple-500 rounded-xl font-bold text-lg transition-colors disabled:opacity-50">
            {modeloActual ? 'Calcular con IA' : 'Esperando modelo...'}
          </button>
        </div>

        {/* Resultado */}
        {resultado && (
          <div className="mt-8 text-center animate-bounce-short">
            <p className="text-gray-400 text-xs uppercase tracking-widest mb-1">Resultado de la Neurona</p>
            <p className="text-5xl font-mono font-bold text-white">{resultado}</p>
          </div>
        )}
      </div>
    </div>
  );
}