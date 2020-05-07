int read_MNIST_train_labels(uint32_t &magic_number_label,
                            uint32_t &number_of_labels, uint8_t *&train_labels);

int read_MNIST_train_images(uint32_t &magic_number_image,
                            uint32_t &number_of_images, uint32_t &n_rows,
                            uint32_t &n_cols, uint32_t &n_pixels,
                            uint32_t &n_features, uint8_t **&train_images);

void extract_train_data_1v1(const uint8_t &class_label_pos,
                            uint32_t &number_of_labels_pos,
                            const uint8_t &class_label_neg,
                            uint32_t &number_of_labels_neg,
                            const uint32_t &number_of_labels,
                            const uint32_t &number_of_images,
                            uint32_t &number_of_data_1v1, uint8_t *train_labels,
                            int8_t *&train_labels_1v1, uint8_t **train_images,
                            uint8_t **&train_images_1v1,
                            const uint32_t &n_features, const uint32_t &n_cols);

int read_MNIST_test_labels(uint32_t &magic_number_label,
                           uint32_t &number_of_labels, uint8_t *&train_labels);

int read_MNIST_test_images(uint32_t &magic_number_image,
                           uint32_t &number_of_images, uint32_t &n_rows,
                           uint32_t &n_cols, uint32_t &n_pixels,
                           uint32_t &n_features, uint8_t **&train_images);

cudaError_t kernel_minibatch_wrapper(int *iters, float *alpha, float *sigma,
                                     float *K, int *y, int l, int C);

cudaError_t kernel_minibatch_block_wrapper(int *iters, float *alpha,
                                           float *sigma, float *K, int *y,
                                           int l, int C);

cudaError_t compute_kernel_matrix(uint32_t n_features,
                                  uint32_t number_of_data_1v1,
                                  uint8_t **train_images_1v1, float *K,
                                  float *B);

cudaError_t compute_kernel_matrix_ts(uint32_t n_features,
                                     uint32_t number_of_data_1v1_ts,
                                     uint8_t **test_images_1v1,
                                     uint32_t number_of_data_1v1,
                                     uint8_t **train_images_1v1, float *K,
                                     float *B);

float dot_product_float(float vect_A[], float vect_B[], int n);

bool kkt_conditions_monitor(uint32_t L, int *y, float *sigma, float *alpha,
                            const int &C);

double mnist_2_class_training(int class1, int class2, int pair_index,
                              uint32_t C, uint32_t M, float accuracy,
                              int *&all_L, float **&all_alpha,
                              uint8_t ***&all_X, int **&all_y);