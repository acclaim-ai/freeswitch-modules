# mod_vad_silero

mod_vad_silero is a FreeSWITCH module that uses the Silero VAD neural network model to detect the start and end points of speech in audio conversations with high accuracy.

## Features

- Real-time Voice Activity Detection using Silero VAD neural network
- Two detection modes: one-shot and continuous
- Configurable speech detection parameters
- SIMD-optimized audio processing (AVX2/SSE2)
- Timestamp-corrected event reporting
- JSON event payloads with speech probability scores

## Installation

1. Download the Silero VAD ONNX model:
   ```bash
   ./download_model.sh
   ```
   Or manually download to `/usr/local/share/silero_vad/silero_vad.onnx`

2. Build and install the module according to your FreeSWITCH build process.

## API

### Commands

#### Start VAD Detection
```
uuid_vad_silero <uuid> start <strategy> <threshold> <silence-ms> <speech-pad-ms> <min-speech-ms> [bugname]
```

**Parameters:**
- `uuid`: FreeSWITCH session UUID (required)
- `strategy`: Detection strategy (required)
  - `one-shot`: Detects speech start, fires event, then stops automatically
  - `continuous`: Continuously monitors and reports both start and stop events
- `threshold`: VAD sensitivity threshold, 0.0-1.0 (required, e.g., 0.5)
- `silence-ms`: Milliseconds of silence needed to detect speech end (required)
- `speech-pad-ms`: Speech padding in milliseconds (required)
- `min-speech-ms`: Minimum speech duration in milliseconds (required)
- `bugname`: Optional media bug name (defaults to "vad_silero")

**Default Values:**
Use `0` for any numeric parameter to apply defaults:
- `silence-ms`: 100ms (when set to 0)
- `speech-pad-ms`: 30ms (when set to 0)
- `min-speech-ms`: 250ms (when set to 0)

#### Stop VAD Detection
```
uuid_vad_silero <uuid> stop [bugname]
```

**Parameters:**
- `uuid`: FreeSWITCH session UUID (required)
- `bugname`: Optional media bug name (defaults to "vad_silero")

### Events

The module fires custom events with subclass `vad_silero:detect`:

#### Event Types
- **Speech Started**: Fired when speech begins
- **Speech Stopped**: Fired when speech ends

#### Event Headers
- `detected-event`: Either "start_talking" or "stop_talking"
- `media-bugname`: The media bug name used

#### Event Body (JSON)
```json
{
  "event": "speech-started|speech-stopped",
  "timestamp": 1234567890123,
  "probability": 0.847
}
```

**Fields:**
- `event`: Event type ("speech-started" or "speech-stopped")
- `timestamp`: Unix timestamp in milliseconds (adjusted for detection delay)
- `probability`: Speech probability score (0.0-1.0)

**Timestamp Adjustment:**
- For speech-started events: timestamp is adjusted backward by `min-speech-ms` to reflect actual speech start
- For speech-stopped events: timestamp is adjusted backward by `silence-ms` to reflect actual speech end

## Usage Examples

### Using FreeSWITCH API
```
# Start continuous detection with default parameters
uuid_vad_silero 12345678-1234-1234-1234-123456789012 start continuous 0.5 0 0 0

# Start one-shot detection with custom parameters  
uuid_vad_silero 12345678-1234-1234-1234-123456789012 start one-shot 0.3 150 40 300 my_vad

# Stop detection
uuid_vad_silero 12345678-1234-1234-1234-123456789012 stop my_vad
```

### Using drachtio-fsmrf
```js
// Start continuous VAD with default parameters
await ep.api('uuid_vad_silero', `${ep.uuid} start continuous 0.5 0 0 0`);

// Start one-shot VAD with custom parameters
await ep.api('uuid_vad_silero', `${ep.uuid} start one-shot 0.3 150 40 300 vad_detect`);

// Stop VAD
await ep.api('uuid_vad_silero', `${ep.uuid} stop vad_detect`);
```

### Event Handling
```js
// Listen for VAD events
ep.on('vad_silero::detect', (evt) => {
  const data = JSON.parse(evt.body);
  console.log(`Speech ${data.event} at ${data.timestamp}, probability: ${data.probability}`);
});
```

## Configuration

### Environment Variables
- `SILERO_VAD_MODEL_PATH`: Custom path to the ONNX model file (default: `/usr/local/share/silero_vad/silero_vad.onnx`)

### Performance Tuning
The module automatically uses SIMD acceleration when available:
- Compile with `-DUSE_AVX2` for AVX2 optimization
- Compile with `-DUSE_SSE2` for SSE2 optimization
- Falls back to scalar processing if neither is available