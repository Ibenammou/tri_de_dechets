//commands line for yousra:
//to navigate to the directory use:
cd C:\Users\Windownet\AppData\Local\Programs\Python\Python39
cd Scripts
//dakchi li ghaykhsek t instali
1-C:\Users\Windownet\AppData\Local\Programs\Python\Python39\python.exe -m pip install imutils
NB: khask tchofi l username dyal pc dyalk o tbdli dik Windownet//
2-C:\Users\Windownet\AppData\Local\Programs\Python\Python39\python.exe -m pip install numpy
3-C:\Users\Windownet\AppData\Local\Programs\Python\Python39\python.exe -m pip install opencv-python
4-C:\Users\Windownet\AppData\Local\Programs\Python\Python39\python.exe -m pip install requests
<<<<<<< HEAD
//for anyone facing running the image classification code u could use this powershell command on ur terminals :
& "C:/Users/Windownet/AppData/Local/Programs/Python/Python39/python.exe" "C:/Users/Windownet/Desktop/waste segregation/tri_de_dechets-1/image_classification.py" --image_dir "C:/Users/Windownet/Desktop/waste segregation/santa/metal_waste" --output_graph "C:/Users/Windownet/Desktop/waste segregation/output_graph_2025-23-05.pb" --output_labels "C:/Users/Windownet/Desktop/waste segregation/output_labels_2025-23-05.txt" --summaries_dir "C:/Users/Windownet/Desktop/waste segregation/summaries" --how_many_training_steps 4000 --learning_rate 0.01 --testing_percentage 10 --validation_percentage 10 --eval_step_interval 10 --train_batch_size 100 --test_batch_size -1 --validation_batch_size 100 --print_misclassified_test_images --model_dir "C:/Users/Windownet/Desktop/waste segregation/imagenet" --bottleneck_dir "C:/Users/Windownet/Desktop/waste segregation/bottleneck" --final_tensor_name "final_result" --flip_left_right --random_crop 0 --random_scale 0 --random_brightness 0
# NB: please make sure u all created the necessary folders and u got ur correct path
def create_inception_graph():
    # Load the Inception model
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Create a new TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Define the input and output tensors for the Inception model
        bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[
                'pool_3/_reshape:0', 'DecodeJpeg/contents:0', 'ResizeBilinear:0']))

        # Modify the input tensor to use placeholder_with_default
        jpeg_data_tensor = tf.compat.v1.placeholder_with_default(
            jpeg_data_tensor, shape=[], name='input/JPEGImage')

    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor
