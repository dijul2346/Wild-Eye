
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

List<CameraDescription>? cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(WildlifeDetectionApp());
}

class WildlifeDetectionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wildlife Detection',
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _cameraController;
  late Interpreter _interpreter;
  bool _isDetecting = false;
  String _detectedAnimal = "";

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  void _initializeCamera() async {
    try {
      _cameraController = CameraController(
        cameras![0],
        ResolutionPreset.medium,
      );
      await _cameraController.initialize();
      if (!mounted) return;
      setState(() {});
      _cameraController.startImageStream((CameraImage image) {
        if (!_isDetecting) {
          _isDetecting = true;
          _runModelOnFrame(image).then((result) {
            setState(() {
              _detectedAnimal = result;
            });
            _isDetecting = false;
          });
        }
      });
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  void _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('best.tflite');
      print('TFLite model loaded');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<String> _runModelOnFrame(CameraImage image) async {
    try {
      final inputImage = _preprocessImage(image);

      List<List<List<double>>> outputBuffer = List.generate(
        1,
        (_) => List.generate(
          25200,
          (_) => List.filled(6, 0.0),
        ),
      );

      _interpreter.run(inputImage.buffer.asUint8List(), outputBuffer);
      return _processOutput(outputBuffer);
    } catch (e) {
      print('Error running model on frame: $e');
      return "Error in detection";
    }
  }

  

TensorImage _preprocessImage(CameraImage image) {
  // Convert CameraImage to a format that can be processed
  // Assuming the image is in YUV420 format
  final int width = image.width;
  final int height = image.height;

  // Create an image buffer
  img.Image convertedImage = img.Image(width, height);

  // Fill the image buffer with pixel data
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Get the Y, U, and V values
      int yIndex = y * image.planes[0].bytesPerRow + x;
      int uIndex = (y ~/ 2) * image.planes[1].bytesPerRow + (x ~/ 2);
      int vIndex = (y ~/ 2) * image.planes[2].bytesPerRow + (x ~/ 2);

      int yValue = image.planes[0].bytes[yIndex];
      int uValue = image.planes[1].bytes[uIndex] - 128;
      int vValue = image.planes[2].bytes[vIndex] - 128;

      // Convert YUV to RGB
      int r = yValue + (1.402 * vValue).round();
      int g = yValue - (0.344136 * uValue + 0.714136 * vValue).round();
      int b = yValue + (1.772 * uValue).round();

      // Clamp values to [0, 255]
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      // Set the pixel in the converted image
      convertedImage.setPixel(x, y, img.getColor(r, g, b));
    }
  }

  // Resize the image to the model input size
  img.Image resizedImage = img.copyResize(convertedImage, width: 640, height: 640);

  // Create a TensorImage object
  TensorImage tensorImage = TensorImage(TfLiteType.uint8);

  // Load the resized image into the TensorImage
  tensorImage.loadImage(resizedImage);

  return tensorImage;
}

  String _processOutput(List<List<List<double>>> outputBuffer) {
    for (var i = 0; i < outputBuffer[0].length; i++) {
      double confidence = outputBuffer[0][i][4];
      if (confidence > 0.5) {
        int classIndex = outputBuffer[0][i][5].toInt();
        switch (classIndex) {
          case 0:
            return "Tiger";
          case 1:
            return "Elephant";
          case 2:
            return "Leopard";
          case 3:
            return "Jaguar";
          case 4:
            return "Cheetah";
        }
      }
    }
    return "No Animal Detected";
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
     
            return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: Text('Wildlife Detection')),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          Positioned(
            bottom: 50,
            left: 20,
            child: Container(
              color: Colors.black.withOpacity(0.5),
              padding: EdgeInsets.all(10),
              child: Text(
                _detectedAnimal,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}