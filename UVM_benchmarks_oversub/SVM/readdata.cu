/*
by Qin Yu, Apr 2019
*/

#include <fstream>
using namespace std;

uint32_t read_little_endian_int(uint32_t little_indian_int) {
  return (little_indian_int >> 24) | (little_indian_int >> 8 & 0x0000FF00) |
         (little_indian_int << 8 & 0x00FF0000) | (little_indian_int << 24);
}

void extract_train_data_1v1(
    const uint8_t &class_label_pos, uint32_t &number_of_labels_pos,
    const uint8_t &class_label_neg, uint32_t &number_of_labels_neg,
    const uint32_t &number_of_labels, const uint32_t &number_of_images,
    uint32_t &number_of_data_1v1, uint8_t *train_labels,
    int8_t *&train_labels_1v1, uint8_t **train_images,
    uint8_t **&train_images_1v1, const uint32_t &n_features,
    const uint32_t &n_cols) {
  for (uint32_t i = 0; i < number_of_labels; i++) {
    if (train_labels[i] == class_label_pos)
      number_of_labels_pos++;
    if (train_labels[i] == class_label_neg)
      number_of_labels_neg++;
  }
  // printf("number of %u = %u, number of %u = %u\n", class_label_pos,
  // number_of_labels_pos, class_label_neg, number_of_labels_neg);
  number_of_data_1v1 = number_of_labels_pos + number_of_labels_neg;

  train_images_1v1 = new uint8_t *[number_of_data_1v1];
  train_labels_1v1 = new int8_t[number_of_data_1v1];
  uint32_t train_1v1_index = 0;
  for (uint32_t i = 0; i < number_of_images; i++) {
    if (train_labels[i] == class_label_pos) {
      train_labels_1v1[train_1v1_index] = 1;
      train_images_1v1[train_1v1_index] = train_images[i];
      train_1v1_index++;
    }
    if (train_labels[i] == class_label_neg) {
      train_labels_1v1[train_1v1_index] = -1;
      train_images_1v1[train_1v1_index] = train_images[i];
      train_1v1_index++;
    }
  }
}

int read_MNIST_train_images(uint32_t &magic_number_image,
                            uint32_t &number_of_images, uint32_t &n_rows,
                            uint32_t &n_cols, uint32_t &n_pixels,
                            uint32_t &n_features, uint8_t **&train_images) {
  ifstream MNIST_train_images_stream("../../data/SVM/train-images.idx3-ubyte",
                                     ios::binary);
  if (MNIST_train_images_stream.is_open()) {
    // printf("train-images.idx3-ubyte is open\n");

    MNIST_train_images_stream.read((char *)&magic_number_image,
                                   sizeof(uint32_t));
    magic_number_image = read_little_endian_int(magic_number_image);
    // printf("magic_number_image = %u\n", magic_number_image);

    MNIST_train_images_stream.read((char *)&number_of_images, sizeof(uint32_t));
    MNIST_train_images_stream.read((char *)&n_rows, sizeof(uint32_t));
    MNIST_train_images_stream.read((char *)&n_cols, sizeof(uint32_t));
    number_of_images = read_little_endian_int(number_of_images);
    n_rows = read_little_endian_int(n_rows);
    n_cols = read_little_endian_int(n_cols);
    n_pixels = n_rows * n_cols;
    n_features = n_pixels + 1;
    // printf("number_of_images = %u, dims = (%u x %u) = %u\nfeatures = %d, size
    // of X = %u\n", number_of_images, n_rows, n_cols, n_pixels, n_features,
    // number_of_images * n_features);

    train_images = new uint8_t *[number_of_images];
    for (uint32_t i = 0; i < number_of_images; i++) {
      train_images[i] = new uint8_t[n_features];
      MNIST_train_images_stream.read((char *)train_images[i], n_pixels);
      train_images[i][n_features - 1] = uint8_t(1);
    }

    MNIST_train_images_stream.close();
  } else {
    printf("train-images.idx3-ubyte not open\n");
    return -1;
  }
  return 0;
}

int read_MNIST_train_labels(uint32_t &magic_number_label,
                            uint32_t &number_of_labels,
                            uint8_t *&train_labels) {
  ifstream MNIST_train_labels_stream("../../data/SVM/train-labels.idx1-ubyte",
                                     ios::binary);
  if (MNIST_train_labels_stream.is_open()) {
    // printf("train-labels.idx1-ubyte is open\n");

    MNIST_train_labels_stream.read((char *)&magic_number_label,
                                   sizeof(uint32_t));
    magic_number_label = read_little_endian_int(magic_number_label);
    // printf("magic_number_label = %u\n", magic_number_label);

    MNIST_train_labels_stream.read((char *)&number_of_labels, sizeof(uint32_t));
    number_of_labels = read_little_endian_int(number_of_labels);
    // printf("number_of_labels = %u\n", number_of_labels);

    train_labels = new uint8_t[number_of_labels];
    // printf("first and last 10 labels:\n");
    for (uint32_t i = 0; i < number_of_labels; i++) {
      MNIST_train_labels_stream.read((char *)&train_labels[i], sizeof(uint8_t));
      // if (i < 10 || i > number_of_labels - 10) printf("%u ",
      // train_labels[i]); if (i == 10) printf("... ");
    } // cout << endl;

    MNIST_train_labels_stream.close();
  } else {
    printf("train-labels.idx1-ubyte not open\n");
    return 1;
  }
  return 0;
}

int read_MNIST_test_images(uint32_t &magic_number_image,
                           uint32_t &number_of_images, uint32_t &n_rows,
                           uint32_t &n_cols, uint32_t &n_pixels,
                           uint32_t &n_features, uint8_t **&train_images) {
  ifstream MNIST_train_images_stream("../../data/SVM/t10k-images.idx3-ubyte",
                                     ios::binary);
  if (MNIST_train_images_stream.is_open()) {
    // printf("t10k-images.idx3-ubyte is open\n");

    MNIST_train_images_stream.read((char *)&magic_number_image,
                                   sizeof(uint32_t));
    magic_number_image = read_little_endian_int(magic_number_image);
    // printf("magic_number_image = %u\n", magic_number_image);

    MNIST_train_images_stream.read((char *)&number_of_images, sizeof(uint32_t));
    MNIST_train_images_stream.read((char *)&n_rows, sizeof(uint32_t));
    MNIST_train_images_stream.read((char *)&n_cols, sizeof(uint32_t));
    number_of_images = read_little_endian_int(number_of_images);
    n_rows = read_little_endian_int(n_rows);
    n_cols = read_little_endian_int(n_cols);
    n_pixels = n_rows * n_cols;
    n_features = n_pixels + 1;
    // printf("number_of_images = %u, dims = (%u x %u) = %u\nfeatures = %d, size
    // of X = %u\n", number_of_images, n_rows, n_cols, n_pixels, n_features,
    // number_of_images * n_features);

    train_images = new uint8_t *[number_of_images];
    for (uint32_t i = 0; i < number_of_images; i++) {
      train_images[i] = new uint8_t[n_features];
      MNIST_train_images_stream.read((char *)train_images[i], n_pixels);
      train_images[i][n_features - 1] = uint8_t(1);
    }

    MNIST_train_images_stream.close();
  } else {
    printf("t10k-images.idx3-ubyte not open\n");
    return 1;
  }
  return 0;
}

int read_MNIST_test_labels(uint32_t &magic_number_label,
                           uint32_t &number_of_labels, uint8_t *&train_labels) {
  ifstream MNIST_train_labels_stream("../../data/SVM/t10k-labels.idx1-ubyte",
                                     ios::binary);
  if (MNIST_train_labels_stream.is_open()) {
    // printf("t10k-labels.idx1-ubyte is open\n");

    MNIST_train_labels_stream.read((char *)&magic_number_label,
                                   sizeof(uint32_t));
    magic_number_label = read_little_endian_int(magic_number_label);
    // printf("magic_number_label = %u\n", magic_number_label);

    MNIST_train_labels_stream.read((char *)&number_of_labels, sizeof(uint32_t));
    number_of_labels = read_little_endian_int(number_of_labels);
    // printf("number_of_labels = %u\n", number_of_labels);

    train_labels = new uint8_t[number_of_labels];
    // printf("first and last 10 labels:\n");
    for (uint32_t i = 0; i < number_of_labels; i++) {
      MNIST_train_labels_stream.read((char *)&train_labels[i], sizeof(uint8_t));
      // if (i < 10 || i > number_of_labels - 10) printf("%u ",
      // train_labels[i]); if (i == 10) printf("... ");
    } // cout << endl;

    MNIST_train_labels_stream.close();
  } else {
    printf("t10k-labels.idx1-ubyte not open\n");
    return 1;
  }
  return 0;
}