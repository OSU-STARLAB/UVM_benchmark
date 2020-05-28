/*
by Qin Yu, Apr 2019
*/

#include <algorithm>  // sort, any_of.
#include <cassert>    // assert.
#include <iostream>   // cout, endl.
using namespace std;

#include "svm.h"

// Not used but may help:
// #include <stdio.h>
// #include <stdlib.h>
// #include <random>
// #include <ctime>

int main(int argc, char const *argv[]) {
  cout << argc << endl;
  for (size_t i = 0; i < argc; i++) {
    cout << "argv[" << i << "] = " << argv[i] << endl;
  }
  if (argc != 4) {
    cout << "Must have 1 integer argument for C and M, and float for ACCURACY" << endl;
    return 1;
  }
  uint32_t C, M;
  float ACCURACY;
  sscanf(argv[1], "%d", &C);
  sscanf(argv[2], "%d", &M);
  sscanf(argv[3], "%f", &ACCURACY);
  cout << "C = " << C << " and M = " << M << " with accuracy = " << ACCURACY << endl;

  int class1_limit = 2;
  int class2_limit = 2;
  int number_of_SVMs = ((class1_limit - 1) * class2_limit) / 2;
  cout << "Will train " << number_of_SVMs << " SVMs" << endl;

  // Record trained SVMs:
  float **all_alpha = new float *[45]();
  uint8_t ***all_X = new uint8_t **[45]();
  int **all_y = new int *[45]();
  int *all_L = new int[45]();
  double *time_all_pairs = new double[45]();
  int pair_counter = 0;
  for (size_t i = 0; i < class1_limit; i++) {
    for (size_t j = 0; j < class2_limit; j++) {
      if (i < j) {
        cout << endl << "Starting training " << i << " vs " << j << endl;
        time_all_pairs[pair_counter] = mnist_2_class_training(
            i, j, pair_counter, C, M, ACCURACY, all_L, all_alpha, all_X, all_y);
        if (time_all_pairs[pair_counter] < 0) {
          cout << "Something Wrong when training SVM " << i << " vs " << j
               << endl;
          return -1;
        }
        // cout << "Time spent on " << i << " vs " << j << " = " <<
        // time_all_pairs[pair_counter] << endl;
        pair_counter++;
      }
    }
  }
  cout << "---------------------------------------------" << endl;
  pair_counter = 0;
  double total_time = 0;
  for (size_t i = 0; i < class1_limit; i++) {
    for (size_t j = 0; j < class2_limit; j++) {
      if (i < j) {
        cout << "Time spent on " << i << " vs " << j << " = "
             << time_all_pairs[pair_counter] << endl;
        total_time += time_all_pairs[pair_counter];
        pair_counter++;
      }
    }
  }
  cout << "Total time spent on training all " << number_of_SVMs
       << "SVMs = " << total_time << endl;

  // Load MNIST test data:
  int mnist_loading_error = 0;
  uint32_t magic_number_label_ts;
  uint32_t number_of_labels_ts;
  uint8_t *test_labels;
  mnist_loading_error = read_MNIST_test_labels(
      magic_number_label_ts, number_of_labels_ts, test_labels);
  if (mnist_loading_error) return -1;

  uint32_t magic_number_image_ts;
  uint32_t number_of_images_ts, n_rows_ts, n_cols_ts, n_pixels_ts,
      n_features_ts;
  uint8_t **test_images;
  mnist_loading_error = read_MNIST_test_images(
      magic_number_image_ts, number_of_images_ts, n_rows_ts, n_cols_ts,
      n_pixels_ts, n_features_ts, test_images);
  if (mnist_loading_error) {
    delete[] test_labels;
    test_labels = NULL;
    for (size_t i = 0; i < number_of_images_ts; i++) {
      delete[] test_images[i];
      test_images[i] = NULL;
    }
    delete[] test_images;
    test_images = NULL;
    return -1;
  }

  // Fraction Table:
  float *B = new float[256 * 256];
  for (int i = 0; i < 256; i++)
    for (int j = 0; j < 256; j++)
      B[i * 256 + j] = pow(i / 255.0f - j / 255.0f, 2);

  // Compute Testing Kernel Matrix:
  int **all_y_hat = new int *[45];
  int L_ts = number_of_images_ts;
  uint8_t **X_ts = test_images;
  uint8_t *y_ts = test_labels;
  cudaError_t cudaStatus;
  int idx = 0;
  for (size_t m = 0; m < class1_limit; m++) {
    for (size_t n = 0; n < class2_limit; n++) {
      if (m < n) {
        int L = all_L[idx];
        uint8_t **X = all_X[idx];
        int *y = all_y[idx];
        all_y_hat[idx] = new int[L_ts];
        float *alpha = all_alpha[idx];
        float *alpha_y = new float[L]();  // alpah .* y
        float *K_ts = new float[L_ts * L];
        cudaStatus = compute_kernel_matrix_ts(785, L_ts, X_ts, L, X, K_ts, B);
        if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "kernel_kernel_matrix launch failed: %s\n",
                  cudaGetErrorString(cudaStatus));
        }
        for (size_t i = 0; i < L; i++) alpha_y[i] = alpha[i] * y[i];
        for (uint32_t i = 0; i < L_ts; i++) {
          float temp_sigma = dot_product_float((K_ts + i * L), alpha_y, L);
          int is_positive = (float(0) < temp_sigma) - (temp_sigma < float(0));
          all_y_hat[idx][i] = (is_positive > 0) ? m : n;
        }

        delete[] alpha_y;
        alpha_y = NULL;
        delete[] K_ts;
        K_ts = NULL;

        idx++;
        // cout << "no problem 1" << endl;
      }  // cout << "no problem 2" << endl;
    }    // cout << "no problem 3" << endl;
  }      // cout << "no problem 4" << endl;

  cout << "L_ts = " << L_ts << endl;
  int *final_pred_y = new int[L_ts];
  int testing_error_number = 0;
  for (size_t i = 0; i < L_ts; i++) {
    int count_vote[10] = {0};
    for (size_t j = 0; j < number_of_SVMs; j++) {
      count_vote[all_y_hat[j][i]] += 1;
    }
    int max_vote_number = 0;
    int max_vote_index = 0;
    for (size_t k = 0; k < 10; k++) {
      if (count_vote[k] > max_vote_number) {
        max_vote_number = count_vote[k];
        max_vote_index = k;
      }
    }
    final_pred_y[i] = max_vote_index;
    if (final_pred_y[i] != y_ts[i]) testing_error_number++;
  }

  float testing_error_rate = float(testing_error_number) / L_ts;
  cout << "Final Testing Error 10 Classes = " << (1 - testing_error_rate) * 100
       << "%" << endl;

  for (size_t i = 0; i < number_of_SVMs; i++) {
    delete[] all_y_hat[i];
    all_y_hat[i] = NULL;
  }
  delete[] all_y_hat;
  all_y_hat = NULL;
  for (size_t i = 0; i < number_of_SVMs; i++) {
    delete[] all_y[i];
    all_y[i] = NULL;
  }
  delete[] all_y;
  all_y = NULL;
  for (size_t i = 0; i < number_of_SVMs; i++) {
    delete[] all_alpha[i];
    all_alpha[i] = NULL;
  }
  delete[] all_alpha;
  all_alpha = NULL;
  for (size_t i = 0; i < number_of_SVMs; i++) {
    for (size_t j = 0; j < all_L[i]; j++) {
      delete[] all_X[i][j];
      all_X[i][j] = NULL;
    }
    delete[] all_X[i];
    all_X[i] = NULL;
  }
  delete[] all_X;
  all_X = NULL;
  delete[] all_L;
  all_L = NULL;
  delete[] time_all_pairs;
  time_all_pairs = NULL;

  delete[] test_labels;
  test_labels = NULL;
  for (size_t i = 0; i < number_of_images_ts; i++) {
    delete[] test_images[i];
    test_images[i] = NULL;
  }
  delete[] test_images;
  test_images = NULL;

  delete[] B;
  B = NULL;
  return 0;
}

double mnist_2_class_training(int class1, int class2, int pair_index,
                              uint32_t C, uint32_t M, float accuracy,
                              int *&all_L, float **&all_alpha,
                              uint8_t ***&all_X, int **&all_y) {
  clock_t total_start_time = clock();
  assert(sizeof(float) == 4);

  // Load MNIST:
  clock_t total_loading_time = clock();
  int mnist_loading_error = 0;

  uint32_t magic_number_label;
  uint32_t number_of_labels;
  uint8_t *train_labels;
  mnist_loading_error = read_MNIST_train_labels(magic_number_label,
                                                number_of_labels, train_labels);
  if (mnist_loading_error) return -1;

  uint32_t magic_number_image;
  uint32_t number_of_images, n_rows, n_cols, n_pixels, n_features;
  uint8_t **train_images;
  mnist_loading_error =
      read_MNIST_train_images(magic_number_image, number_of_images, n_rows,
                              n_cols, n_pixels, n_features, train_images);
  if (mnist_loading_error) {
    delete[] train_labels;
    train_labels = NULL;
    for (size_t i = 0; i < number_of_images; i++) {
      delete[] train_images[i];
      train_images[i] = NULL;
    }
    delete[] train_images;
    train_images = NULL;
    return -1;
  }
  clock_t time_finished_loading = clock();
  cout << "load time = "
       << double(time_finished_loading - total_start_time) / CLOCKS_PER_SEC
       << endl;

  // Extract 1v1 data:
  uint8_t class_label_pos = uint8_t(class1), class_label_neg = uint8_t(class2);
  uint32_t number_of_labels_pos = 0, number_of_labels_neg = 0,
           number_of_data_1v1 = 0;
  uint8_t **train_images_1v1;
  int8_t *train_labels_1v1;
  extract_train_data_1v1(class_label_pos, number_of_labels_pos, class_label_neg,
                         number_of_labels_neg, number_of_labels,
                         number_of_images, number_of_data_1v1, train_labels,
                         train_labels_1v1, train_images, train_images_1v1,
                         n_features, n_cols);
  clock_t time_finished_extracting = clock();
  cout << "extract time = "
       << double(time_finished_extracting - time_finished_loading) /
              CLOCKS_PER_SEC
       << endl;

  // Fraction Table:
  float *K = new float[number_of_data_1v1 *
                       number_of_data_1v1];  // Must be defined by `new`.
  float *B = new float[256 * 256];
  for (int i = 0; i < 256; i++)
    for (int j = 0; j < 256; j++)
      B[i * 256 + j] = pow(i / 255.0f - j / 255.0f, 2);

  // Compute Kernel Matrix:
  cudaError_t cudaStatus;
  clock_t time_started_kernelmatrix = clock();
  cudaStatus = compute_kernel_matrix(n_features, number_of_data_1v1,
                                     train_images_1v1, K, B);

  clock_t time_finished_kernelmatrix = clock();
  cout << "matrix time = "
       << double(time_finished_kernelmatrix - time_started_kernelmatrix) /
              CLOCKS_PER_SEC
       << endl;

  // Set meta parameters:
  const uint32_t L = number_of_data_1v1;

  // Define alpha, sigma:
  int *iters = new int[1]();
  float *alpha = new float[L]();  // `()` initialises it to zeros.
  float *sigma = new float[L]();
  int *y = new int[L];
  for (size_t i = 0; i < L; i++) y[i] = train_labels_1v1[i];
  float *alpha_y = new float[L]();  // alpah .* y

  // Select initial working set:
  uint32_t *all_data_point_idx = new uint32_t[L];
  for (uint32_t i = 0; i < M; i++) all_data_point_idx[i] = i;
  uint32_t *support_vector_idx = new uint32_t[M];
  for (uint32_t i = 0; i < M; i++) support_vector_idx[i] = i;
  uint32_t number_of_sv = M;

  // Decomposition/Mini-batch algorithm:
  uint32_t kkt_counter = 0;
  // int number_of_single_violation = 0;
  double total_minibatch_optimisation_time = 0;
  uint32_t remaining_number_of_sv = 0;
  while (!kkt_conditions_monitor(L, y, sigma, alpha, C) && kkt_counter < 100) {
    kkt_counter++;
    cout << "LOOPING: " << kkt_counter << " ";

    // Select data for GPU (not prepare GPU data):
    int *mini_y = new int[number_of_sv];
    float *mini_a = new float[number_of_sv];
    float *mini_s = new float[number_of_sv];
    float *mini_K = new float[number_of_sv * number_of_sv];
    for (size_t i = 0; i < number_of_sv; i++) {
      mini_y[i] = y[support_vector_idx[i]];
      mini_a[i] = alpha[support_vector_idx[i]];
      mini_s[i] = sigma[support_vector_idx[i]];
      for (size_t j = 0; j < number_of_sv; j++)
        mini_K[i * number_of_sv + j] =
            K[support_vector_idx[i] * L + support_vector_idx[j]];
    }

    // Call GPU kernel (including GPU data preparation):
    clock_t kernel_start_time = clock();
    if (number_of_sv > 1024) {
      cudaStatus = kernel_minibatch_wrapper(iters, mini_a, mini_s, mini_K,
                                            mini_y, number_of_sv, C);
    } else {
      cudaStatus = kernel_minibatch_block_wrapper(iters, mini_a, mini_s, mini_K,
                                                  mini_y, number_of_sv, C);
    }
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "wrapper failed!");
      return -1;
    }
    clock_t kernel_finish_time = clock();
    total_minibatch_optimisation_time +=
        double(kernel_finish_time - kernel_start_time) / CLOCKS_PER_SEC;

    // Update gradient:
    for (size_t i = 0; i < number_of_sv; i++)
      alpha[support_vector_idx[i]] = mini_a[i];
    for (size_t i = 0; i < L; i++) {
      sigma[i] = 0;
      for (size_t j = 0; j < L; j++) sigma[i] += alpha[j] * y[j] * K[i * L + j];
    }

    // Remove non-support vectors:
    // To avoid getting right array length by another loop, allocate excessive
    // amount of memory.
    uint32_t *remaining_support_vector_idx = new uint32_t[number_of_sv];
    remaining_number_of_sv = 0;
    for (size_t i = 0; i < number_of_sv; i++) {
      if (mini_a[i] == 0) continue;
      remaining_support_vector_idx[remaining_number_of_sv] =
          support_vector_idx[i];
      remaining_number_of_sv++;
    }
    delete[] support_vector_idx;
    support_vector_idx = new uint32_t[remaining_number_of_sv];
    memcpy(support_vector_idx, remaining_support_vector_idx,
           remaining_number_of_sv * sizeof(uint32_t));
    delete[] remaining_support_vector_idx;
    remaining_support_vector_idx = NULL;

    // Select new points who violate KKT conditions:
    float *violation_val_dirty = new float[L];
    uint32_t *violation_idx_dirty = new uint32_t[L];
    uint32_t number_of_violations = 0;
    const float kkt1 = 1 - accuracy;
    const float kkt2 = 1 + accuracy;
    const float kkt3 = 0.01;
    for (uint32_t i = 0; i < L; i++) {
      float yfx = y[i] * sigma[i];
      if ((alpha[i] == 0 && yfx < kkt1) ||
          (alpha[i] == C && yfx > kkt2) ||
          (0 < alpha[i] && alpha[i] < C && !(abs(yfx - 1) < kkt3))) {
        violation_idx_dirty[number_of_violations] = i;
        violation_val_dirty[number_of_violations] = yfx;
        number_of_violations++;
      }
    }
    cout << "number of new violation is " << number_of_violations << endl;
    float *violation_val = new float[number_of_violations];
    uint32_t *violation_idx = new uint32_t[number_of_violations];
    memcpy(violation_val, violation_val_dirty,
           number_of_violations * sizeof(float));
    memcpy(violation_idx, violation_idx_dirty,
           number_of_violations * sizeof(uint32_t));
    delete[] violation_val_dirty;
    violation_val_dirty = NULL;
    delete[] violation_idx_dirty;
    violation_idx_dirty = NULL;

    // Sort new points and discard duplication (with respect to remaining
    // working set):
    uint32_t *sort_perm = new uint32_t[number_of_violations];
    for (uint32_t i = 0; i < number_of_violations; i++) sort_perm[i] = i;
    sort(sort_perm, sort_perm + number_of_violations,
         [violation_val](uint32_t a, uint32_t b) -> bool {
           return violation_val[a] < violation_val[b];
         });
    uint32_t *violation_idx_sorted_unique = new uint32_t[number_of_violations];
    uint32_t number_of_violations_unique = 0;
    for (size_t i = 0; i < number_of_violations; i++) {
      uint32_t this_support_vector_idx = violation_idx[sort_perm[i]];
      if (any_of(support_vector_idx,
                 support_vector_idx + remaining_number_of_sv,
                 [this_support_vector_idx](uint32_t idx) {
                   return idx == this_support_vector_idx;
                 }))
        continue;
      violation_idx_sorted_unique[number_of_violations_unique] =
          this_support_vector_idx;
      number_of_violations_unique++;
    }
    delete[] sort_perm;
    sort_perm = NULL;
    delete[] violation_val;
    violation_val = NULL;
    delete[] violation_idx;
    violation_idx = NULL;

    // Concatenate remaining working set and violation set:
    number_of_sv = remaining_number_of_sv + number_of_violations_unique;
    uint32_t *new_support_vector_idx = new uint32_t[number_of_sv];
    memcpy(new_support_vector_idx, support_vector_idx,
           remaining_number_of_sv * sizeof(uint32_t));
    memcpy(new_support_vector_idx + remaining_number_of_sv,
           violation_idx_sorted_unique,
           number_of_violations_unique * sizeof(uint32_t));
    delete[] violation_idx_sorted_unique;
    violation_idx_sorted_unique = NULL;
    delete[] support_vector_idx;
    support_vector_idx = new_support_vector_idx;

    // Delete dynamic allocated memories:
    delete[] mini_y;
    mini_y = NULL;
    delete[] mini_a;
    mini_a = NULL;
    delete[] mini_s;
    mini_s = NULL;
    delete[] mini_K;
    mini_K = NULL;
  }
  delete[] support_vector_idx;
  support_vector_idx = NULL;
  cout << "Remaining Number of SV = " << remaining_number_of_sv << endl;

  // Predict training set and get training error:
  // clock_t training_error_time_start = clock();
  uint32_t training_error_number = 0;
  for (uint32_t i = 0; i < L; i++) {
    int y_hat = (float(0) < sigma[i]) - (sigma[i] < float(0));
    if (y_hat == y[i]) continue;
    training_error_number++;
  }
  float training_error_rate = float(training_error_number) / L;
  cout << "Precision = " << (1 - training_error_rate) * 100 << "%; ";
  // clock_t training_error_time_finish = clock();
  // double training_error_time = double(training_error_time_finish -
  // training_error_time_start) / CLOCKS_PER_SEC; cout << "Precision time cost =
  // " << training_error_time << endl;

  // Record alpha, X and y:
  all_L[pair_index] = L;
  all_alpha[pair_index] = new float[L];
  all_y[pair_index] = new int[L];
  all_X[pair_index] = new uint8_t *[L];
  for (size_t i = 0; i < L; i++) {
    all_alpha[pair_index][i] = alpha[i];
    all_y[pair_index][i] = y[i];
    all_X[pair_index][i] = new uint8_t[n_features];
    for (size_t j = 0; j < n_features; j++) {
      all_X[pair_index][i][j] = train_images_1v1[i][j];
    }
  }

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return -1;
  }

  delete[] train_labels;
  train_labels = NULL;
  for (size_t i = 0; i < number_of_images; i++) {
    delete[] train_images[i];
    train_images[i] = NULL;
  }
  delete[] train_images;
  train_images = NULL;
  delete[] train_images_1v1;
  train_images_1v1 = NULL;
  delete[] train_labels_1v1;
  train_labels_1v1 = NULL;
  delete[] B;
  B = NULL;
  delete[] K;
  K = NULL;
  // delete[] K_ts; K_ts = NULL;
  delete[] alpha;
  alpha = NULL;
  delete[] sigma;
  sigma = NULL;
  delete[] iters;
  iters = NULL;
  // No need to delete duplications: y.

  clock_t total_finish_time = clock();
  double total_time =
      double(total_finish_time - total_start_time) / CLOCKS_PER_SEC;
  cout << "Total svm time used = " << total_minibatch_optimisation_time << endl;
  cout << "Total time used = " << total_time << endl;
  return total_time;
}

bool kkt_conditions_monitor(uint32_t L, int *y, float *sigma, float *alpha,
                            const int &C) {
  for (uint32_t i = 0; i < L; i++) {
    float yfx = y[i] * sigma[i];
    if (alpha[i] == 0 && yfx < 0.99) {
      // printf("$1 the %u th alpha %.7f has yf(x) = %f\n", i, alpha[i], yfx);
      return false;
    } else if (alpha[i] == C && yfx > 1.01) {
      // printf("$2 the %u th alpha %.7f has yf(x) = %f\n", i, alpha[i], yfx);
      return false;
    } else if (0 < alpha[i] && alpha[i] < C && !(abs(yfx - 1) < 0.01)) {
      // printf("$3 the %u th alpha %.7f has yf(x) = %f\n", i, alpha[i], yfx);
      return false;  // Don't 0 < alpha[i] < C, Wrong!
    }
  }
  return true;
}

float dot_product_float(float vect_A[], float vect_B[], int n) {
  float product = 0;
  for (int i = 0; i < n; i++) product += vect_A[i] * vect_B[i];
  return product;
}
