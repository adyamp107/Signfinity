window.resizeTo(1200, 700);

// ================================================================================
// element
// ================================================================================
// navigator element

let navigation_menu_home_logo_picture = document.querySelector('.navigation_menu_home_logo_picture');
let navigation_menu_dataset_logo_picture = document.querySelector('.navigation_menu_dataset_logo_picture');
let navigation_menu_training_logo_picture = document.querySelector('.navigation_menu_training_logo_picture');
let navigation_menu_translate_logo_picture = document.querySelector('.navigation_menu_translate_logo_picture');
let navigation_menu_history_logo_picture = document.querySelector('.navigation_menu_history_logo_picture');
let navigation_menu_about_us_logo_picture = document.querySelector('.navigation_menu_about_us_logo_picture');
let navigation_menu_setting_logo_picture = document.querySelector('.navigation_menu_setting_logo_picture');

let button_3 = document.querySelectorAll('.button_3');
let page_1 = document.querySelectorAll('.page_1');

// ================================================================================
// home element

let button_7 = document.querySelectorAll('.button_7');

// ================================================================================
// dataset element

let dataset_control_type_select = document.querySelector('.dataset_control_type_select');
let dataset_control_select_path_button = document.querySelector('.dataset_control_select_path_button');
let dataset_control_create_button = document.querySelector('.dataset_control_create_button');
let dataset_control_delete_button = document.querySelector('.dataset_control_delete_button');

let dataset_selected_path_file_folder_text = document.querySelector('.dataset_selected_path_file_folder_text');
let dataset_selected_path_clear_button = document.querySelector('.dataset_selected_path_clear_button');

let dataset_word_list_sub_frame = document.querySelector('.dataset_word_list_sub_frame');

let dataset_selected_word_label_text = document.querySelector('.dataset_selected_word_label_text');
let dataset_selected_word_clear_button = document.querySelector('.dataset_selected_word_clear_button');

let dataset_edit_add_segmented_button = document.querySelector('.dataset_edit_add_segmented_button');
let dataset_edit_delete_segmented_button = document.querySelector('.dataset_edit_delete_segmented_button');
let dataset_edit_redata_segmented_button = document.querySelector('.dataset_edit_redata_segmented_button');
let dataset_edit_relabel_segmented_button = document.querySelector('.dataset_edit_relabel_segmented_button');
let button_2 = document.querySelectorAll('.button_2');

let dataset_edit_add_sub_frame = document.querySelector('.dataset_edit_add_sub_frame');
let dataset_edit_delete_sub_frame = document.querySelector('.dataset_edit_delete_sub_frame');
let dataset_edit_redata_sub_frame = document.querySelector('.dataset_edit_redata_sub_frame');
let dataset_edit_relabel_sub_frame = document.querySelector('.dataset_edit_relabel_sub_frame');
let frame_2 = document.querySelectorAll('.frame_2');

let dataset_edit_add_new_word_text = document.querySelector('.dataset_edit_add_new_word_text');
let dataset_edit_add_take_button = document.querySelector('.dataset_edit_add_take_button');
let add_camera_frame = document.querySelector('.add_camera_frame');
let add_camera_video = document.querySelector('.add_camera_video');
let add_control_word_text = document.querySelector('.add_control_word_text');
let add_control_start_button = document.querySelector('.add_control_start_button');
let add_control_stop_button = document.querySelector('.add_control_stop_button');
let add_control_marker_check = document.querySelector('.add_control_marker_check');
let add_progress_progressbar = document.querySelector('.add_progress_progressbar');
let add_progress_information_text = document.querySelector('.add_progress_information_text');

let dataset_edit_delete_word_button = document.querySelector('.dataset_edit_delete_word_button');

let dataset_edit_redata_word_button = document.querySelector('.dataset_edit_redata_word_button');
let redata_camera_frame = document.querySelector('.redata_camera_frame');
let redata_camera_video = document.querySelector('.redata_camera_video');
let redata_control_word_text = document.querySelector('.redata_control_word_text');
let redata_control_start_button = document.querySelector('.redata_control_start_button');
let redata_control_stop_button = document.querySelector('.redata_control_stop_button');
let redata_control_marker_check = document.querySelector('.redata_control_marker_check');
let redata_progress_progressbar = document.querySelector('.redata_progress_progressbar');
let redata_progress_information_text = document.querySelector('.redata_progress_information_text');

let dataset_edit_relabel_new_word_text = document.querySelector('.dataset_edit_relabel_new_word_text');
let dataset_edit_relabel_change_button = document.querySelector('.dataset_edit_relabel_change_button');

// ================================================================================
// training element
let training_control_select_path_button = document.querySelector('.training_control_select_path_button');
let training_control_algorithm_select = document.querySelector('.training_control_algorithm_select');
let training_control_train_button = document.querySelector('.training_control_train_button');
let training_control_delete_button = document.querySelector('.training_control_delete_button');
let training_selected_path_file_folder_text = document.querySelector('.training_selected_path_file_folder_text');
let training_selected_path_clear_button = document.querySelector('.training_selected_path_clear_button');

let training_graph_classification_report_segmented_button = document.querySelector('.training_graph_classification_report_segmented_button');
let training_graph_confusion_matrix_segmented_button = document.querySelector('.training_graph_confusion_matrix_segmented_button');
let training_graph_error_rate_segmented_button = document.querySelector('.training_graph_error_rate_segmented_button');
let training_graph_epoch_loss_segmented_button = document.querySelector('.training_graph_epoch_loss_segmented_button');

let training_graph_classification_report_image = document.querySelector('.training_graph_classification_report_image');
let training_graph_classification_report_select_path_button = document.querySelector('.training_graph_classification_report_select_path_button');
let training_graph_classification_report_selected_path_text = document.querySelector('.training_graph_classification_report_selected_path_text');
let training_graph_classification_report_name_text = document.querySelector('.training_graph_classification_report_name_text');
let training_graph_classification_report_save_button = document.querySelector('.training_graph_classification_report_save_button');

let training_table_classification_report_sub_frame = document.querySelector('.training_table_classification_report_sub_frame');
let training_table_classification_report_select_path_button = document.querySelector('.training_table_classification_report_select_path_button');
let training_table_classification_report_selected_path_text = document.querySelector('.training_table_classification_report_selected_path_text');
let training_table_classification_report_name_text = document.querySelector('.training_table_classification_report_name_text');
let training_table_classification_report_save_button = document.querySelector('.training_table_classification_report_save_button');

let training_graph_confusion_matrix_image = document.querySelector('.training_graph_confusion_matrix_image');
let training_graph_confusion_matrix_select_path_button = document.querySelector('.training_graph_confusion_matrix_select_path_button');
let training_graph_confusion_matrix_selected_path_text = document.querySelector('.training_graph_confusion_matrix_selected_path_text');
let training_graph_confusion_matrix_name_text = document.querySelector('.training_graph_confusion_matrix_name_text');
let training_graph_confusion_matrix_save_button = document.querySelector('.training_graph_confusion_matrix_save_button');

let training_table_confusion_matrix_sub_frame = document.querySelector('.training_table_confusion_matrix_sub_frame');
let training_table_confusion_matrix_select_path_button = document.querySelector('.training_table_confusion_matrix_select_path_button');
let training_table_confusion_matrix_selected_path_text = document.querySelector('.training_table_confusion_matrix_selected_path_text');
let training_table_confusion_matrix_name_text = document.querySelector('.training_table_confusion_matrix_name_text');
let training_table_confusion_matrix_save_button = document.querySelector('.training_table_confusion_matrix_save_button');

let training_graph_error_rate_image = document.querySelector('.training_graph_error_rate_image');
let training_graph_error_rate_select_path_button = document.querySelector('.training_graph_error_rate_select_path_button');
let training_graph_error_rate_selected_path_text = document.querySelector('.training_graph_error_rate_selected_path_text');
let training_graph_error_rate_name_text = document.querySelector('.training_graph_error_rate_name_text');
let training_graph_error_rate_save_button = document.querySelector('.training_graph_error_rate_save_button');

let training_table_error_rate_sub_frame = document.querySelector('.training_table_error_rate_sub_frame');
let training_table_error_rate_select_path_button = document.querySelector('.training_table_error_rate_select_path_button');
let training_table_error_rate_selected_path_text = document.querySelector('.training_table_error_rate_selected_path_text');
let training_table_error_rate_name_text = document.querySelector('.training_table_error_rate_name_text');
let training_table_error_rate_save_button = document.querySelector('.training_table_error_rate_save_button');

let training_graph_epoch_loss_image = document.querySelector('.training_graph_epoch_loss_image');
let training_graph_epoch_loss_select_path_button = document.querySelector('.training_graph_epoch_loss_select_path_button');
let training_graph_epoch_loss_selected_path_text = document.querySelector('.training_graph_epoch_loss_selected_path_text');
let training_graph_epoch_loss_name_text = document.querySelector('.training_graph_epoch_loss_name_text');
let training_graph_epoch_loss_save_button = document.querySelector('.training_graph_epoch_loss_save_button');

let training_table_epoch_loss_sub_frame = document.querySelector('.training_table_epoch_loss_sub_frame');
let training_table_epoch_loss_select_path_button = document.querySelector('.training_table_epoch_loss_select_path_button');
let training_table_epoch_loss_selected_path_text = document.querySelector('.training_table_epoch_loss_selected_path_text');
let training_table_epoch_loss_name_text = document.querySelector('.training_table_epoch_loss_name_text');
let training_table_epoch_loss_save_button = document.querySelector('.training_table_epoch_loss_save_button');

let training_graph_epoch_accuracy_image = document.querySelector('.training_graph_epoch_accuracy_image');
let training_graph_epoch_accuracy_select_path_button = document.querySelector('.training_graph_epoch_accuracy_select_path_button');
let training_graph_epoch_accuracy_selected_path_text = document.querySelector('.training_graph_epoch_accuracy_selected_path_text');
let training_graph_epoch_accuracy_name_text = document.querySelector('.training_graph_epoch_accuracy_name_text');
let training_graph_epoch_accuracy_save_button = document.querySelector('.training_graph_epoch_accuracy_save_button');

let training_table_epoch_accuracy_sub_frame = document.querySelector('.training_table_epoch_accuracy_sub_frame');
let training_table_epoch_accuracy_select_path_button = document.querySelector('.training_table_epoch_accuracy_select_path_button');
let training_table_epoch_accuracy_selected_path_text = document.querySelector('.training_table_epoch_accuracy_selected_path_text');
let training_table_epoch_accuracy_name_text = document.querySelector('.training_table_epoch_accuracy_name_text');
let training_table_epoch_accuracy_save_button = document.querySelector('.training_table_epoch_accuracy_save_button');

// ================================================================================
// translate element

let translate_camera_frame = document.querySelector('.translate_camera_frame');
let translate_camera_video = document.querySelector('.translate_camera_video');

let translate_control_result_text = document.querySelector('.translate_control_result_text');
let translate_control_time_limit_progressbar = document.querySelector('.translate_control_time_limit_progressbar');
let translate_control_marker_button = document.querySelector('.translate_control_marker_button');
let translate_control_restart_button = document.querySelector('.translate_control_restart_button');
let translate_control_start_button = document.querySelector('.translate_control_start_button');
let translate_control_stop_button = document.querySelector('.translate_control_stop_button');
let translate_control_erase_button = document.querySelector('.translate_control_erase_button');

let translate_translation_list_sub_frame = document.querySelector('.translate_translation_list_sub_frame');
let translate_translation_list_no_data_title = document.querySelector('.translate_translation_list_no_data_title');
// ================================================================================
// history element

let history_control_select_path_button = document.querySelector('.history_control_select_path_button');
let history_control_create_button = document.querySelector('.history_control_create_button');
let history_control_delete_button = document.querySelector('.history_control_delete_button');

let history_selected_path_file_folder_text = document.querySelector('.history_selected_path_file_folder_text');
let history_selected_path_clear_button = document.querySelector('.history_selected_path_clear_button');

let history_date_time_list_sub_frame = document.querySelector('.history_date_time_list_sub_frame');

let history_selected_date_time_label_text = document.querySelector('.history_selected_date_time_label_text');
let history_selected_date_time_clear_button = document.querySelector('.history_selected_date_time_clear_button');
let history_selected_date_time_delete_button = document.querySelector('.history_selected_date_time_delete_button');

let history_translation_list_sub_frame = document.querySelector('.history_translation_list_sub_frame');

// ================================================================================
// setting element
let setting_app_control_color_mode_select = document.querySelector('.setting_app_control_color_mode_select');
let setting_app_control_color_theme_select = document.querySelector('.setting_app_control_color_theme_select');

let setting_landmark_control_pose_check = document.querySelector('.setting_landmark_control_pose_check');
let setting_landmark_control_face_check = document.querySelector('.setting_landmark_control_face_check');
let setting_landmark_control_right_hand_check = document.querySelector('.setting_landmark_control_right_hand_check');
let setting_landmark_control_left_hand_check = document.querySelector('.setting_landmark_control_left_hand_check');

let setting_bounding_box_control_pose_check = document.querySelector('.setting_bounding_box_control_pose_check');
let setting_bounding_box_control_face_check = document.querySelector('.setting_bounding_box_control_face_check');
let setting_bounding_box_control_right_hand_check = document.querySelector('.setting_bounding_box_control_right_hand_check');
let setting_bounding_box_control_left_hand_check = document.querySelector('.setting_bounding_box_control_left_hand_check');

let button_4 = document.querySelectorAll('.button_4');
let frame_3 = document.querySelectorAll('.frame_3');

// ================================================================================
// all form

let form_create_dataset = null;
let form_create_history = null;
let form_create_training_random_forest = null;
let form_create_training_decision_tree = null;
let form_create_training_k_nearest_neighbors = null;
let form_create_training_convolutional_neural_network = null;

let form_create_training_support_vector_machine = null;
let form_create_training_naive_bayes = null;

// ================================================================================
// camera
eel.expose(update_image);
function update_image(image_data) {
    add_camera_video.src = 'data:image/jpeg;base64,' + image_data;
    redata_camera_video.src = 'data:image/jpeg;base64,' + image_data;
    translate_camera_video.src = 'data:image/jpeg;base64,' + image_data;
}
eel.camera_event();

// ================================================================================
// navigator
button_3.forEach((button_3_1) => {
    button_3_1.addEventListener('click', () => {
        eel.check_navigator()(function(checked) {
            if(checked) {
                button_3.forEach((button_3_2) => {
                    button_3_2.classList.remove('selected_button_3');
                });
                button_3_1.classList.add('selected_button_3');        
                page_1.forEach((page_1_1) => {
                    if(page_1_1.classList[0].split('_')[0] == button_3_1.classList[0].split('_')[2]) {
                        page_1_1.classList.add('selected_page_1');
                        if(page_1_1.classList[0].split('_')[0] == 'translate') {
                            eel.navigator_to_translate_page_event();
                        } else {
                            eel.navigator_to_other_than_translate_page_event();
                        }
                    } else {
                        page_1_1.classList.remove('selected_page_1');
                    }
                });
            }
        });
    });
});

// ================================================================================
// home

button_7.forEach((button_7_1) => {
    button_7_1.addEventListener('click', () => {
        button_name = button_7_1.classList[0].split('_')[1];
        console.log(button_name)
        button_3.forEach((button_3_1) => {
            if(button_3_1.classList[0].split('_')[2] == button_name) {
                console.log(button_name)
                button_3_1.click();

            }
        });
    });
});

// ================================================================================
// dataset

// control dataset
function clear_dataset_selected_path_file_folder_text_event() {
    dataset_selected_path_file_folder_text.value = '';
    clear_dataset_word_list_sub_frame_event();
    clear_dataset_selected_word_label_text_event();
}

function clear_dataset_word_list_sub_frame_event() {
    while(dataset_word_list_sub_frame.firstChild) {
        dataset_word_list_sub_frame.removeChild(dataset_word_list_sub_frame.firstChild);
    }
    let divElement = document.createElement('div');
    divElement.classList.add('dataset_word_list_no_data_title', 'title_1');
    divElement.textContent = 'No Data';
    dataset_word_list_sub_frame.appendChild(divElement);
}

function clear_dataset_selected_word_label_text_event() {
    dataset_selected_word_label_text.value = '';
}

function dataset_selected_path_file_folder_text_event(dataset_path) {
    if(dataset_path) {
        eel.dataset_selected_path_file_folder_text_event(dataset_path, dataset_control_type_select.value)(function(dataset_data) {
            if(dataset_data) {
                clear_dataset_selected_path_file_folder_text_event();
                dataset_selected_path_file_folder_text.value = dataset_data.dataset_path;
                if(dataset_data.words.length > 0) {
                    while(dataset_word_list_sub_frame.firstChild) {
                        dataset_word_list_sub_frame.removeChild(dataset_word_list_sub_frame.firstChild);
                    }
                    dataset_data.words.forEach((word) => {
                        let divElement = document.createElement('div');
                        new_class = 'dataset_word_list_' + word + '_button';
                        new_class = new_class.replace(/\s+/g, '#');
                        divElement.classList.add(new_class, 'button_5');
                        divElement.textContent = word;
                        dataset_word_list_sub_frame.appendChild(divElement);
                        divElement.addEventListener('click', () => {
                            dataset_selected_word_label_text.value = divElement.textContent;
                        });
                    });
                }
            }
        });
    }
}

dataset_control_type_select.addEventListener('change', () => {
    clear_dataset_selected_path_file_folder_text_event();
    if(dataset_control_type_select.value == 'Landmark File') {
        training_control_algorithm_select.innerHTML = `
            <option value='Random Forest'>Random Forest</option>
            <option value='K-Nearest Neighbors'>K-Nearest Neighbors</option>
            <option value='Decision Tree'>Decision Tree</option>
            <option value="Support Vector Machine">Support Vector Machine</option>
            <option value="Naive Bayes">Naive Bayes</option>
        `;
    } else if (dataset_control_type_select.value == 'Image Folder') {
        training_control_algorithm_select.innerHTML = `
            <option value='Convolutional Neural Network'>Convolutional Neural Network</option>
            <option value='Azure Machine Learning (CNN)'>Azure Machine Learning (CNN)</option>
        `;
    }
});

dataset_control_select_path_button.addEventListener('click', () => {
    eel.dataset_control_select_path_button_event(dataset_control_type_select.value)(function(dataset_path) {
        dataset_selected_path_file_folder_text_event(dataset_path);
    });
});

dataset_control_create_button.addEventListener('click', () => {
    form_create_dataset = window.open('./form/create_dataset.html', 'Form Window', 'width=500,height=500');
});

dataset_control_delete_button.addEventListener('click', () => {
    eel.dataset_control_delete_button_event(dataset_selected_path_file_folder_text.value)(function(confirmation_delete) {
        if(confirmation_delete) {
            clear_dataset_selected_path_file_folder_text_event();
        }
    });
});

// selected path
dataset_selected_path_clear_button.addEventListener('click', () => {
    clear_dataset_selected_path_file_folder_text_event();
});

// selected word
dataset_selected_word_clear_button.addEventListener('click', () => {
    clear_dataset_selected_word_label_text_event();
});

// edit dataset
button_2.forEach((button_2_1) => {
    button_2_1.addEventListener('click', () => {
        button_2.forEach((button_2_2) => {
            button_2_2.classList.remove('selected_button_2');
        });
        button_2_1.classList.add('selected_button_2');
        frame_2.forEach((frame) => {
            if(frame.classList[0].split('_')[2] == button_2_1.classList[0].split('_')[2]) {
                frame.classList.add('selected_frame_2');
            } else {
                frame.classList.remove('selected_frame_2');
            }
        });
    });
});

// edit dataset (add)
dataset_edit_add_take_button.addEventListener('click', () => {
    eel.dataset_edit_add_take_button_event(dataset_selected_path_file_folder_text.value, dataset_edit_add_new_word_text.value, dataset_control_type_select.value)(function(new_word) {
        if(new_word) {
            add_control_word_text.value = new_word;
            page_1.forEach((page_1_1) => {
                if(page_1_1.classList[0].split('_')[0] == 'add') {
                    page_1_1.classList.add('selected_page_1');
                } else {
                    page_1_1.classList.remove('selected_page_1');
                }
            });   
        }
    });
});
add_control_marker_check.addEventListener('change', () => {
    eel.add_control_marker_check_event(add_control_marker_check.checked);
});
add_control_start_button.addEventListener('click', () => {
    eel.add_control_start_button_event(dataset_selected_path_file_folder_text.value, add_control_word_text.value, dataset_control_type_select.value);
});
add_control_stop_button.addEventListener('click', () => {
    eel.add_control_stop_button_event();
});

eel.expose(update_add_progress);
function update_add_progress(percentage, count_data) {
    add_progress_progressbar.value = percentage;
    add_progress_information_text.innerHTML = 'Progress: ' + percentage.toFixed(2) + '%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data Amount: ' + count_data + ' Data';
}

// edit dataset (delete)
dataset_edit_delete_word_button.addEventListener('click', () => {
    eel.dataset_edit_delete_word_button_event(dataset_selected_path_file_folder_text.value, dataset_selected_word_label_text.value, dataset_control_type_select.value);
});

// edit dataset (redata)
dataset_edit_redata_word_button.addEventListener('click', () => {
    eel.dataset_edit_redata_word_button_event(dataset_selected_path_file_folder_text.value, dataset_selected_word_label_text.value)(function(word) {
        if(word) {
            redata_control_word_text.value = word;
            page_1.forEach((page_1_1) => {
                if(page_1_1.classList[0].split('_')[0] == 'redata') {
                    page_1_1.classList.add('selected_page_1');
                } else {
                    page_1_1.classList.remove('selected_page_1');
                }
            });
        }
    });
});

redata_control_marker_check.addEventListener('change', () => {
    eel.redata_control_marker_check_event(redata_control_marker_check.checked);
});

redata_control_start_button.addEventListener('click', () => {
    eel.redata_control_start_button_event(dataset_selected_path_file_folder_text.value, dataset_selected_word_label_text.value, dataset_control_type_select.value);
});

redata_control_stop_button.addEventListener('click', () => {
    eel.redata_control_stop_button_event();
});

eel.expose(update_redata_progress);
function update_redata_progress(percentage, count_data) {
    redata_progress_progressbar.value = percentage;
    redata_progress_information_text.innerHTML = 'Progress: ' + percentage.toFixed(2) + '%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data Amount: ' + count_data + ' Data';
}

// edit dataset (relabel)
dataset_edit_relabel_change_button.addEventListener('click', () => {
    eel.dataset_edit_relabel_change_button_event(dataset_selected_path_file_folder_text.value, dataset_selected_word_label_text.value, dataset_edit_relabel_new_word_text.value, dataset_control_type_select.value);
});

eel.expose(to_dataset_page);
function to_dataset_page() {
    add_control_word_text.value = '';
    redata_control_word_text.value = '';
    page_1.forEach((page_1_1) => {
        if(page_1_1.classList[0].split('_')[0] == 'dataset') {
            page_1_1.classList.add('selected_page_1');
        } else {
            page_1_1.classList.remove('selected_page_1');
        }
    });
    dataset_selected_path_file_folder_text_event(dataset_selected_path_file_folder_text.value)
}

// ================================================================================
// training
function clear_training_selected_path_file_folder_text() {
    training_selected_path_file_folder_text.value = '';
    training_graph_classification_report_image.src = '';
    training_graph_confusion_matrix_image.src = '';
    training_graph_error_rate_image.src = '';
    training_graph_epoch_loss_image.src = '';
    training_table_classification_report_sub_frame.innerHTML = `
        <div class="training_table_classification_report_no_data">No Data</div>
    `;
    training_table_confusion_matrix_sub_frame.innerHTML = `
        <div class="training_table_confusion_matrix_no_data">No Data</div>
    `;
    training_table_error_rate_sub_frame.innerHTML = `
        <div class="training_table_error_rate_no_data">No Data</div>
    `;
    training_table_epoch_loss_sub_frame.innerHTML = `
        <div class="training_table_epoch_loss_no_data">No Data</div>
    `;
    training_table_epoch_accuracy_sub_frame.innerHTML = `
        <div class="training_table_epoch_accuracy_no_data">No Data</div>
    `;
}

training_control_select_path_button.addEventListener('click', () => {
    eel.training_control_select_path_button_event();
});

training_control_train_button.addEventListener('click', () => {
    eel.training_control_train_button_event(dataset_selected_path_file_folder_text.value)(function(check_dataset) {
        training_algorithm = training_control_algorithm_select.value;
        if(check_dataset) {
            if(training_algorithm == 'Convolutional Neural Network') {
                form_create_training_convolutional_neural_network = window.open('./form/create_convolutional_neural_network.html', 'Form Window', 'width=500,height=500');
            } else if(training_algorithm == 'Azure Machine Learning (CNN)') {
                eel.azure_machine_learning_cnn()();
            } else if(training_algorithm == 'Random Forest') {
                form_create_training_random_forest = window.open('./form/create_random_forest.html', 'Form Window', 'width=500,height=650');
            } else if(training_algorithm == 'K-Nearest Neighbors') {
                form_create_training_k_nearest_neighbors = window.open('./form/create_k_nearest_neighbors.html', 'Form Window', 'width=500,height=650');            
            } else if(training_algorithm == 'Decision Tree') {
                form_create_training_decision_tree = window.open('./form/create_decision_tree.html', 'Form Window', 'width=500,height=600');
            } else if(training_algorithm == 'Support Vector Machine') {
                form_create_training_support_vector_machine = window.open('./form/create_support_vector_machine.html', 'Form Window', 'width=500,height=600');
            } else if(training_algorithm == 'Naive Bayes') {
                form_create_training_naive_bayes = window.open('./form/create_naive_bayes.html', 'Form Window', 'width=500,height=600');
            }
        }
    });
});

training_control_delete_button.addEventListener('click', () => {
    eel.training_control_delete_button_event(training_selected_path_file_folder_text.value)(function(check_delete) {
        if(check_delete) {
            clear_training_selected_path_file_folder_text();
        }
    });
});

training_selected_path_clear_button.addEventListener('click', () => {
    clear_training_selected_path_file_folder_text();
});

button_4.forEach((button_4_1) => {
    button_4_1.addEventListener('click', () => {
        button_4.forEach((button_4_2) => {
            button_4_2.classList.remove('selected_button_4');
        });
        button_4_1.classList.add('selected_button_4');
        frame_3.forEach((frame) => {
            if((frame.classList[0].split('_')[2] + frame.classList[0].split('_')[3]) == (button_4_1.classList[0].split('_')[2] + button_4_1.classList[0].split('_')[3])) {
                frame.classList.add('selected_frame_3');
            } else {
                frame.classList.remove('selected_frame_3');
            }
        });
    });
});

eel.expose(training_selected_path_file_folder_text_event);
function training_selected_path_file_folder_text_event(training_path) {
    clear_training_selected_path_file_folder_text();
    training_selected_path_file_folder_text.value = training_path;
}

eel.expose(update_training_graph);
function update_training_graph(trained_model, graph, graphic_type) {
    if(graphic_type == 'classification_report') {
        training_graph_classification_report_image.src = 'data:image/jpeg;base64,' + graph;
        row_table = ``;
        for(row in trained_model['classification_report']) {
            if(row == 'accuracy') {
                row_data = `
                <tr>
                    <td> </td>                    
                    <td> </td>
                    <td> </td>
                    <td> </td>
                    <td> </td>
                </tr>
                <tr>
                    <td>${row}</td>                    
                    <td> </td>
                    <td> </td>
                    <td>${trained_model['classification_report'][row]}</td>
                    <td>${trained_model['classification_report']['macro avg']['support']}</td>
                </tr>
                `;
                row_table += row_data;
            } else {
                row_data = `
                <tr>
                    <td>${row}</td>                    
                    <td>${trained_model['classification_report'][row]['precision']}</td>
                    <td>${trained_model['classification_report'][row]['recall']}</td>
                    <td>${trained_model['classification_report'][row]['f1-score']}</td>
                    <td>${trained_model['classification_report'][row]['support']}</td>
                </tr>
                `;
                row_table += row_data;
            }
        }
        training_table_classification_report_sub_frame.innerHTML = `
        <table>
            <tr>
                <th>Label</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
            ${row_table}
        </table>
        `;
    } else if(graphic_type == 'confusion_matrix') {
        training_graph_confusion_matrix_image.src = 'data:image/jpeg;base64,' + graph;
        row_table = ``;
        for(row_index in trained_model['confusion_matrix']) {
            row_data = `
            <tr>
            `;
            for(column_index in trained_model['confusion_matrix'][row_index]) {
                if(row_index == column_index) {
                    row_data += `
                        <td class='diagonal_confusion_matrix'>${trained_model['confusion_matrix'][row_index][column_index]}</td>
                    `;
                } else {
                    row_data += `
                        <td>${trained_model['confusion_matrix'][row_index][column_index]}</td>
                    `;
                }
            }
            row_data += `
            </tr>
            `;
            row_table += row_data;
        }
        training_table_confusion_matrix_sub_frame.innerHTML = `
        <table>
            ${row_table}
        </table>
        `;
    } else if(graphic_type == 'error_rate') {
        training_graph_error_rate_image.src = 'data:image/jpeg;base64,' + graph;
        row_table = ``;
        check_oob_header = ``;
        console.log(trained_model)
        for(row_index in trained_model['error_rate']['range']) {
            row_table += `
            <tr>
                <td>${trained_model['error_rate']['range'][row_index]}</td>
                <td>${trained_model['error_rate']['train_error'][row_index]}</td>
                <td>${trained_model['error_rate']['test_error'][row_index]}</td>
            `;
            if(trained_model['error_rate']['oob_error']) {
                check_oob_header = `<th>OOB Error</th>`;
                row_table += `
                    <td>${trained_model['error_rate']['oob_error'][row_index]}</td>
                `;
            }
            row_table += `
            </tr>
            `;
        }
        training_table_error_rate_sub_frame.innerHTML = `
        <table>
            <tr>
                <th>Range</th>
                <th>Train Error</th>
                <th>Test Error</th>
                ${check_oob_header}
            </tr>
            ${row_table}
        </table>
        `;
    } else if(graphic_type == 'epoch_loss') {
        training_graph_epoch_loss_image.src = 'data:image/jpeg;base64,' + graph;
        row_table = ``;
        for(row_index in trained_model['epoch_loss']['epoch']) {
            row_table += `
            <tr>
                <td>${trained_model['epoch_loss']['epoch'][row_index]}</td>
                <td>${trained_model['epoch_loss']['training_loss'][row_index]}</td>
                <td>${trained_model['epoch_loss']['validation_loss'][row_index]}</td>
            </tr>
            `;
        }
        training_table_epoch_loss_sub_frame.innerHTML = `
        <table>
            <tr>
                <th>Epochs</th>
                <th>Training Loss</th>
                <th>Validation Loss</th>
            </tr>
            ${row_table}
        </table>
        `;
    } else if(graphic_type == 'epoch_accuracy') {
        console.log(trained_model)
        training_graph_epoch_accuracy_image.src = 'data:image/jpeg;base64,' + graph;
        row_table = ``;
        for(row_index in trained_model['epoch_accuracy']['epoch']) {
            row_table += `
            <tr>
                <td>${trained_model['epoch_accuracy']['epoch'][row_index]}</td>
                <td>${trained_model['epoch_accuracy']['training_accuracy'][row_index]}</td>
                <td>${trained_model['epoch_accuracy']['validation_accuracy'][row_index]}</td>
            </tr>
            `;
        }
        training_table_epoch_accuracy_sub_frame.innerHTML = `
        <table>
            <tr>
                <th>Epochs</th>
                <th>Training Accuracy</th>
                <th>Validation Accuracy</th>
            </tr>
            ${row_table}
        </table>
        `;
    }
}

// ================================================================================
// translate

eel.expose(clear_translate_translation_list_sub_frame_event)
function clear_translate_translation_list_sub_frame_event() {
    translate_translation_list_sub_frame.innerHTML = `
        <div class="translate_translation_list_no_data_title title_1">No Data</div>               
    `;
    translate_translation_list_no_data_title = document.querySelector('.translate_translation_list_no_data_title');
}

eel.expose(translate_control_result_text_event);
function translate_control_result_text_event() {
    return translate_control_result_text.value;
}

eel.expose(send_text_translate_control_result_text_event)
function send_text_translate_control_result_text_event(text_output) {
    translate_control_result_text.value = text_output;
}

eel.expose(translate_control_time_limit_progressbar_event);
function translate_control_time_limit_progressbar_event(count_time) {
    translate_control_time_limit_progressbar.value = count_time;
}

eel.expose(send_translate_translation_list_sub_frame_event)
function send_translate_translation_list_sub_frame_event(history_data) {
    if(translate_translation_list_sub_frame.contains(translate_translation_list_no_data_title)) {
        translate_translation_list_sub_frame.removeChild(translate_translation_list_no_data_title);
    }
    translate_translation_list_sub_frame.innerHTML += `
        <div>Date: ${history_data['date']} | Time: ${history_data['time']}</div>
        <br>
        <div>${history_data['translate']}</div>
        <br>
        <br>
    `;
}

translate_control_marker_button.addEventListener('click', () => {
    eel.translate_control_marker_button_event()(function(check_marker) {
        if(check_marker) {
            translate_control_marker_button.classList.add('selected_button_6');
        } else {
            translate_control_marker_button.classList.remove('selected_button_6');
        }
    });
});

translate_control_restart_button.addEventListener('click', () => {
    eel.empty_check_translate()();
    translate_control_result_text.value = '';
    translate_control_time_limit_progressbar.value = 0;
    translate_control_time_limit_progressbar.value = 0;
    translate_control_time_limit_progressbar.value = 0;
    translate_control_time_limit_progressbar.value = 0;
    translate_control_time_limit_progressbar.value = 0;
});

translate_control_start_button.addEventListener('click', () => {
    eel.translate_control_start_button_event(training_selected_path_file_folder_text.value, history_selected_path_file_folder_text.value);
});

translate_control_stop_button.addEventListener('click', () => {
    eel.translate_control_stop_button_event()(function(update_history) {
        if(update_history) {
            eel.history_selected_path_file_folder_text_event(update_history)(function(history_data) {
                history_selected_path_file_folder_text_event(update_history, history_data);
            })
        }
    });
});

translate_control_erase_button.addEventListener('click', () => {
    eel.empty_check_translate()();
    translate_control_result_text.value = translate_control_result_text.value.trim().split(' ').slice(0, -1).join(' ') + ' ';
    if(translate_control_result_text.value == ' ') {
        translate_control_result_text.value = '';
        translate_control_time_limit_progressbar.value = 0;
        translate_control_time_limit_progressbar.value = 0;
        translate_control_time_limit_progressbar.value = 0;
        translate_control_time_limit_progressbar.value = 0;
        translate_control_time_limit_progressbar.value = 0;
    }
});

// ================================================================================
// history

function clear_history_selected_path_file_folder_text() {
    history_selected_path_file_folder_text.value = '';
    history_date_time_list_sub_frame.innerHTML = `
        <div class="history_date_time_list_no_data_title title_1">No Data</div>
    `;
    clear_history_selected_date_time_label_text();
}

function clear_history_selected_date_time_label_text() {
    history_selected_date_time_label_text.value = '';
    history_translation_list_sub_frame.innerHTML = `
        <div class="history_translation_list_no_data_title title_1">No Data</div>               
    `;
}

function history_selected_path_file_folder_text_event(history_path, history_data) {
    if(history_data) {
        clear_history_selected_path_file_folder_text();
        history_selected_path_file_folder_text.value = history_path;
        if(history_data.length > 0) {
            while(history_date_time_list_sub_frame.firstChild) {
                history_date_time_list_sub_frame.removeChild(history_date_time_list_sub_frame.firstChild);
            }
            for(sheet_index in history_data) {
                let divElement = document.createElement('div');
                divElement.classList.add('button_5');
                divElement.textContent = history_data[sheet_index]['sheet'];
                history_date_time_list_sub_frame.appendChild(divElement);
                divElement.addEventListener('click', () => {
                    history_selected_date_time_label_text.value = divElement.textContent;
                    history_translation_list_sub_frame.innerHTML = ``;              
                    console.log(history_data[history_data.findIndex(item => item['sheet'] === history_selected_date_time_label_text.value)])
                    for(values_index in history_data[history_data.findIndex(item => item['sheet'] === history_selected_date_time_label_text.value)]['values']) {
                        history_translation_list_sub_frame.innerHTML += `
                            <div>Date: ${history_data[history_data.findIndex(item => item['sheet'] === history_selected_date_time_label_text.value)]['values'][values_index][1]} | Time: ${history_data[history_data.findIndex(item => item['sheet'] === history_selected_date_time_label_text.value)]['values'][values_index][2]}</div>
                            <br>
                            <div>${history_data[history_data.findIndex(item => item['sheet'] === history_selected_date_time_label_text.value)]['values'][values_index][0]}</div>
                            <br>
                            <br>
                        `;
                    }
                });
            }
        }
    }
}

history_control_select_path_button.addEventListener('click', () => {
    eel.history_control_select_path_button_event()(function(history_path) {
        eel.history_selected_path_file_folder_text_event(history_path)(function(history_data) {
            history_selected_path_file_folder_text_event(history_path, history_data);
        })
    });
});

history_control_create_button.addEventListener('click', () => {
    form_create_history = window.open('./form/create_history.html', 'Form Window', 'width=500,height=500');
});

history_control_delete_button.addEventListener('click', () => {
    eel.history_control_delete_button_event(history_selected_path_file_folder_text.value)(function(check_delete) {
        if(check_delete) {
            clear_history_selected_path_file_folder_text();
        }
    });
});

history_selected_path_clear_button.addEventListener('click', () => {
    clear_history_selected_path_file_folder_text();
});

history_selected_date_time_clear_button.addEventListener('click', () => {
    clear_history_selected_date_time_label_text();
});

history_selected_date_time_delete_button.addEventListener('click', () => {
    eel.history_selected_date_time_delete_button_event(history_selected_path_file_folder_text.value, history_selected_date_time_label_text.value.replace(' | ', ' ').replace(/:/g, '_'))(function(check_delete) {
        if(check_delete == 'delete_sheet') {
            eel.history_selected_path_file_folder_text_event(history_selected_path_file_folder_text.value)(function(history_data) {
                history_selected_path_file_folder_text_event(history_selected_path_file_folder_text.value, history_data);
            });
        } else if(check_delete == 'delete_file') {
            clear_history_selected_path_file_folder_text();
        }
    });
});

// ================================================================================
// setting
setting_landmark_control_face_check.addEventListener('change', () => {
    eel.setting_landmark_control_face_check_event(setting_landmark_control_face_check.checked);
});
setting_landmark_control_pose_check.addEventListener('change', () => {
    eel.setting_landmark_control_pose_check_event(setting_landmark_control_pose_check.checked);
});
setting_landmark_control_left_hand_check.addEventListener('change', () => {
    eel.setting_landmark_control_left_hand_check_event(setting_landmark_control_left_hand_check.checked);
});
setting_landmark_control_right_hand_check.addEventListener('change', () => {
    eel.setting_landmark_control_right_hand_check_event(setting_landmark_control_right_hand_check.checked);
});
setting_bounding_box_control_face_check.addEventListener('change', () => {
    eel.setting_bounding_box_control_face_check_event(setting_bounding_box_control_face_check.checked);
});
setting_bounding_box_control_pose_check.addEventListener('change', () => {
    eel.setting_bounding_box_control_pose_check_event(setting_bounding_box_control_pose_check.checked);
});
setting_bounding_box_control_left_hand_check.addEventListener('change', () => {
    eel.setting_bounding_box_control_left_hand_check_event(setting_bounding_box_control_left_hand_check.checked);
});
setting_bounding_box_control_right_hand_check.addEventListener('change', () => {
    eel.setting_bounding_box_control_right_hand_check_event(setting_bounding_box_control_right_hand_check.checked);
});

const light_color_mode_root = {
    '--color-gray-1': 'rgb(245, 245, 245)',
    '--color-gray-2': 'rgb(195, 195, 195)',
    '--color-gray-3': 'rgb(205, 205, 205)',
    '--color-gray-4': 'rgb(215, 215, 215)',
    '--color-gray-5': 'rgb(225, 225, 225)',
    '--color-main': 'rgb(0, 0, 0)'
};

const dark_color_mode_root = {
    '--color-gray-1': 'rgb(10, 10, 10)',
    '--color-gray-2': 'rgb(40, 40, 40)',
    '--color-gray-3': 'rgb(50, 50, 50)',
    '--color-gray-4': 'rgb(60, 60, 60)',
    '--color-gray-5': 'rgb(70, 70, 70)',
    '--color-main': 'rgb(225, 225, 225)'
};

setting_app_control_color_mode_select_event('System');
function setting_app_control_color_mode_select_event(color_mode) {
    switch(color_mode) {
        case 'System':
            let dark_mode = window.matchMedia('(prefers-color-scheme: dark)');
            if (dark_mode.matches) {
                // dark
                for(let color in dark_color_mode_root) {
                    document.documentElement.style.setProperty(color, dark_color_mode_root[color]);
                    localStorage.setItem(color, dark_color_mode_root[color]);
                }
                navigation_menu_home_logo_picture.src = './assets/light_home.png';
                navigation_menu_dataset_logo_picture.src = './assets/light_dataset.png';
                navigation_menu_training_logo_picture.src = './assets/light_training.png';
                navigation_menu_translate_logo_picture.src = './assets/light_translate.png';
                navigation_menu_history_logo_picture.src = './assets/light_history.png';
                navigation_menu_about_us_logo_picture.src = './assets/light_about_us.png';
                navigation_menu_setting_logo_picture.src = './assets/light_setting.png';
            } else {
                // light
                for(let color in light_color_mode_root) {
                    document.documentElement.style.setProperty(color, light_color_mode_root[color]);
                    localStorage.setItem(color, light_color_mode_root[color]);
                }
                navigation_menu_home_logo_picture.src = './assets/dark_home.png';
                navigation_menu_dataset_logo_picture.src = './assets/dark_dataset.png';
                navigation_menu_training_logo_picture.src = './assets/dark_training.png';
                navigation_menu_translate_logo_picture.src = './assets/dark_translate.png';
                navigation_menu_history_logo_picture.src = './assets/dark_history.png';
                navigation_menu_about_us_logo_picture.src = './assets/dark_about_us.png';
                navigation_menu_setting_logo_picture.src = './assets/dark_setting.png';
            }
            break;
        case 'Light':
            for(let color in light_color_mode_root) {
                document.documentElement.style.setProperty(color, light_color_mode_root[color]);
                localStorage.setItem(color, light_color_mode_root[color]);
            }
            navigation_menu_home_logo_picture.src = './assets/dark_home.png';
            navigation_menu_dataset_logo_picture.src = './assets/dark_dataset.png';
            navigation_menu_training_logo_picture.src = './assets/dark_training.png';
            navigation_menu_translate_logo_picture.src = './assets/dark_translate.png';
            navigation_menu_history_logo_picture.src = './assets/dark_history.png';
            navigation_menu_about_us_logo_picture.src = './assets/dark_about_us.png';
            navigation_menu_setting_logo_picture.src = './assets/dark_setting.png';
            break;
        case 'Dark':
            for(let color in dark_color_mode_root) {
                document.documentElement.style.setProperty(color, dark_color_mode_root[color]);
                localStorage.setItem(color, dark_color_mode_root[color]);
            }
            navigation_menu_home_logo_picture.src = './assets/light_home.png';
            navigation_menu_dataset_logo_picture.src = './assets/light_dataset.png';
            navigation_menu_training_logo_picture.src = './assets/light_training.png';
            navigation_menu_translate_logo_picture.src = './assets/light_translate.png';
            navigation_menu_history_logo_picture.src = './assets/light_history.png';
            navigation_menu_about_us_logo_picture.src = './assets/light_about_us.png';
            navigation_menu_setting_logo_picture.src = './assets/light_setting.png';  
            break;                        
    }
    if(form_create_dataset) {
        form_create_dataset.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_history) {
        form_create_history.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_random_forest) {
        form_create_training_random_forest.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_decision_tree) {
        form_create_training_decision_tree.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_k_nearest_neighbors) {
        form_create_training_k_nearest_neighbors.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_convolutional_neural_network) {
        form_create_training_convolutional_neural_network.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_support_vector_machine) {
        form_create_training_support_vector_machine.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
    if(form_create_training_naive_bayes) {
        form_create_training_naive_bayes.postMessage({
            element_event: 'setting_app_control_color_mode_select_event'
        }, '*');
    }
}

const blue_color_theme_root = {
    '--color-blue-1': 'rgb(0, 160, 230)',
    '--color-blue-2': 'rgb(0, 130, 230)',
    '--color-blue-3': 'rgb(0, 100, 230)'
};
const green_color_theme_root = {
    '--color-blue-1': 'rgb(0, 230, 160)',
    '--color-blue-2': 'rgb(0, 200, 130)',
    '--color-blue-3': 'rgb(0, 170, 100)'
};
const yellow_color_theme_root = {
    '--color-blue-1': 'rgb(195, 195, 0)',
    '--color-blue-2': 'rgb(165, 165, 0)',
    '--color-blue-3': 'rgb(135, 135, 0)'
};
const pink_color_theme_root = {
    '--color-blue-1': 'rgb(230, 0, 160)',
    '--color-blue-2': 'rgb(200, 0, 130)',
    '--color-blue-3': 'rgb(170, 0, 100)'
};
const violet_color_theme_root = {
    '--color-blue-1': 'rgb(160, 0, 230)',
    '--color-blue-2': 'rgb(130, 0, 200)',
    '--color-blue-3': 'rgb(100, 0, 170)'
};
const orange_color_theme_root = {
    '--color-blue-1': 'rgb(230, 160, 0)',
    '--color-blue-2': 'rgb(200, 130, 0)',
    '--color-blue-3': 'rgb(170, 100, 0)'
};
setting_app_control_color_theme_select_event('Blue');
function setting_app_control_color_theme_select_event(color_theme) {
    switch(color_theme) {
        case 'Blue':
            for (let color in blue_color_theme_root) {
                document.documentElement.style.setProperty(color, blue_color_theme_root[color]);
                localStorage.setItem(color, blue_color_theme_root[color]);
            }
            break;
        case 'Green':
            for (let color in green_color_theme_root) {
                document.documentElement.style.setProperty(color, green_color_theme_root[color]);
                localStorage.setItem(color, green_color_theme_root[color]);
            }      
            break;
        case 'Yellow':
            for (let color in yellow_color_theme_root) {
                document.documentElement.style.setProperty(color, yellow_color_theme_root[color]);
                localStorage.setItem(color, yellow_color_theme_root[color]);
            }      
            break;
        case 'Pink':
            for (let color in pink_color_theme_root) {
                document.documentElement.style.setProperty(color, pink_color_theme_root[color]);
                localStorage.setItem(color, pink_color_theme_root[color]);
            }   
            break;
        case 'Violet':
            for (let color in violet_color_theme_root) {
                document.documentElement.style.setProperty(color, violet_color_theme_root[color]);
                localStorage.setItem(color, violet_color_theme_root[color]);
            }      
            break;                                                
        case 'Orange':
            for (let color in orange_color_theme_root) {
                document.documentElement.style.setProperty(color, orange_color_theme_root[color]);
                localStorage.setItem(color, orange_color_theme_root[color]);
            }
            break;
    }
    if(form_create_dataset) {
        form_create_dataset.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_history) {
        form_create_history.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_random_forest) {
        form_create_training_random_forest.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_decision_tree) {
        form_create_training_decision_tree.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_k_nearest_neighbors) {
        form_create_training_k_nearest_neighbors.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_convolutional_neural_network) {
        form_create_training_convolutional_neural_network.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_support_vector_machine) {
        form_create_training_support_vector_machine.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
    if(form_create_training_naive_bayes) {
        form_create_training_naive_bayes.postMessage({
            element_event: 'setting_app_control_color_theme_select_event'
        }, '*');
    }
}

setting_app_control_color_mode_select_event('System');
setting_app_control_color_mode_select.addEventListener('change', () => {
    setting_app_control_color_mode_select_event(setting_app_control_color_mode_select.value);
});
setting_app_control_color_theme_select.addEventListener('change', () => {
    setting_app_control_color_theme_select_event(setting_app_control_color_theme_select.value);
});

// ================================================================================
// on close

function close_all_form() {
    if(form_create_dataset != null) {
        form_create_dataset.close();
    }
    if(form_create_history != null) {
        form_create_history.close();
    }
    if(form_create_training_random_forest != null) {
        form_create_training_random_forest.close();
    }
    if(form_create_training_decision_tree != null) {
        form_create_training_decision_tree.close();
    }
    if(form_create_training_k_nearest_neighbors != null) {
        form_create_training_k_nearest_neighbors.close();
    }
    if(form_create_training_convolutional_neural_network != null) {
        form_create_training_convolutional_neural_network.close();
    }
    if(form_create_training_support_vector_machine != null) {
        form_create_training_support_vector_machine.close();
    }
    if(form_create_training_naive_bayes != null) {
        form_create_training_naive_bayes.close();
    }
}

window.addEventListener('beforeunload', () => {
    close_all_form();
});

// ================================================================================
// form
window.addEventListener('message', (event) => {
    if(event.origin !== window.location.origin) return;
    // dataset
    if(event.data.element_event == 'form_create_dataset_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(dataset_path) {
            if(form_create_dataset) {
                form_create_dataset.postMessage({
                    element_event: 'form_create_dataset_selected_path_text',
                    dataset_path: dataset_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_dataset_create_button') {
        eel.dataset_control_create_button_event(event.data.dataset_path, event.data.dataset_name, dataset_control_type_select.value)(function(dataset_path) {
            if(dataset_path) {
                dataset_selected_path_file_folder_text_event(dataset_path);
                close_all_form();
            }
        });
    }
    // history
    if(event.data.element_event == 'form_create_history_select_path_button') {
        eel.form_history_control_select_path_button_event()(function(history_path) {
            if(form_create_history) {
                form_create_history.postMessage({
                    element_event: 'form_create_history_selected_path_text',
                    history_path: history_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_history_create_button') {
        eel.history_control_create_button_event(event.data.history_path, event.data.history_name)(function(history_path) {
            if(history_path) {
                eel.history_selected_path_file_folder_text_event(history_path)(function(history_data) {
                    history_selected_path_file_folder_text_event(history_path, history_data);
                });                
                close_all_form();
            }
        });
    }
    // decision tree
    else if(event.data.element_event == 'form_create_training_decision_tree_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_decision_tree) {
                form_create_training_decision_tree.postMessage({
                    element_event: 'form_create_training_decision_tree_selected_path_text',
                    training_decision_tree_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_decision_tree_create_button') {
        eel.form_create_training_decision_tree_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_decision_tree_path, event.data.training_decision_tree_name, event.data.training_decision_tree_random_state, event.data.training_decision_tree_max_depth)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
    // random forest
    else if(event.data.element_event == 'form_create_training_random_forest_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_random_forest) {
                form_create_training_random_forest.postMessage({
                    element_event: 'form_create_training_random_forest_selected_path_text',
                    training_random_forest_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_random_forest_create_button') {
        eel.form_create_training_random_forest_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_random_forest_path, event.data.training_random_forest_name, event.data.training_random_forest_random_state, event.data.training_random_forest_n_estimators)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
    // k nearest neighbors
    else if(event.data.element_event == 'form_create_training_k_nearest_neighbors_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_k_nearest_neighbors) {
                form_create_training_k_nearest_neighbors.postMessage({
                    element_event: 'form_create_training_k_nearest_neighbors_selected_path_text',
                    training_k_nearest_neighbors_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_k_nearest_neighbors_create_button') {
        eel.form_create_training_k_nearest_neighbors_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_k_nearest_neighbors_path, event.data.training_k_nearest_neighbors_name, event.data.training_k_nearest_neighbors_random_state, event.data.training_k_nearest_neighbors_n_neighbors)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
    // convolutional neural network
    else if(event.data.element_event == 'form_create_training_convolutional_neural_network_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_convolutional_neural_network) {
                form_create_training_convolutional_neural_network.postMessage({
                    element_event: 'form_create_training_convolutional_neural_network_selected_path_text',
                    training_convolutional_neural_network_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_convolutional_neural_network_create_button') {
        eel.form_create_training_convolutional_neural_network_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_convolutional_neural_network_path, event.data.training_convolutional_neural_network_name, event.data.training_convolutional_neural_network_shuffle, event.data.training_convolutional_neural_network_epochs)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
    // support vector machine
    else if(event.data.element_event == 'form_create_training_support_vector_machine_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_support_vector_machine) {
                form_create_training_support_vector_machine.postMessage({
                    element_event: 'form_create_training_support_vector_machine_selected_path_text',
                    training_support_vector_machine_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_support_vector_machine_create_button') {
        eel.form_create_training_support_vector_machine_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_support_vector_machine_path, event.data.training_support_vector_machine_name, event.data.training_support_vector_machine_random_state, event.data.training_support_vector_machine_c, event.data.training_support_vector_machine_kernel)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
    // naive bayes
    else if(event.data.element_event == 'form_create_training_naive_bayes_select_path_button') {
        eel.dataset_control_select_path_button_event('Image Folder')(function(training_path) {
            if(form_create_training_naive_bayes) {
                form_create_training_naive_bayes.postMessage({
                    element_event: 'form_create_training_naive_bayes_selected_path_text',
                    training_naive_bayes_path: training_path
                }, '*');
            }
        });
    } else if(event.data.element_event == 'form_create_training_naive_bayes_create_button') {
        eel.form_create_training_naive_bayes_create_button_event(dataset_selected_path_file_folder_text.value, event.data.training_naive_bayes_path, event.data.training_naive_bayes_name, event.data.training_naive_bayes_random_state, event.data.training_naive_bayes_var_smoothing, event.data.training_naive_bayes_step)(function(training_path) {
            if(training_path) {
                eel.training_graph_event(training_path);
                close_all_form();
            }
        });
    }
});

eel.expose(update_decision_tree_progress);
function update_decision_tree_progress(percentage, max_depth) {
    form_create_training_decision_tree.postMessage({
        element_event: 'form_create_training_decision_tree_progress_progressbar',
        percentage: percentage,
        max_depth: max_depth
    }, '*');
}

eel.expose(update_random_forest_progress);
function update_random_forest_progress(percentage, n_estimators) {
    form_create_training_random_forest.postMessage({
        element_event: 'form_create_training_random_forest_progress_progressbar',
        percentage: percentage,
        n_estimators: n_estimators
    }, '*');
}

eel.expose(update_k_nearest_neighbors_progress);
function update_k_nearest_neighbors_progress(percentage, n_neighbors) {
    form_create_training_k_nearest_neighbors.postMessage({
        element_event: 'form_create_training_k_nearest_neighbors_progress_progressbar',
        percentage: percentage,
        n_neighbors: n_neighbors
    }, '*');
}

eel.expose(update_support_vector_machine_progress);
function update_support_vector_machine_progress(percentage, c) {
    form_create_training_support_vector_machine.postMessage({
        element_event: 'form_create_training_support_vector_machine_progress_progressbar',
        percentage: percentage,
        c: c
    }, '*');
}

eel.expose(update_naive_bayes_progress);
function update_naive_bayes_progress(percentage, var_smoothing) {
    form_create_training_naive_bayes.postMessage({
        element_event: 'form_create_training_naive_bayes_progress_progressbar',
        percentage: percentage,
        var_smoothing: var_smoothing
    }, '*');
}

// ================================================================================
// resize
window.addEventListener('resize', () => {
    let add_width_frame = add_camera_frame.getBoundingClientRect().width;
    let add_height_frame= add_camera_frame.getBoundingClientRect().height;
    let add_width_based_on_height = (add_height_frame * 4) / 3;
    let add_height_based_on_width = (add_width_frame * 3) / 4;
    if(add_width_based_on_height <= add_width_frame) {
        add_camera_video.style.width = add_width_based_on_height + 'px';
    } else {
        add_camera_video.style.width = add_width_frame + 'px';
    }
    if(add_height_based_on_width <= add_height_frame) {
        add_camera_video.style.height = add_height_based_on_width + 'px';
    } else {
        add_camera_video.style.height = add_height_frame + 'px';
    }

    let redata_width_frame = redata_camera_frame.getBoundingClientRect().width;
    let redata_height_frame= redata_camera_frame.getBoundingClientRect().height;
    let redata_width_based_on_height = (redata_height_frame * 4) / 3;
    let redata_height_based_on_width = (redata_width_frame * 3) / 4;
    if(redata_width_based_on_height <= redata_width_frame) {
        redata_camera_video.style.width = redata_width_based_on_height + 'px';
    } else {
        redata_camera_video.style.width = redata_width_frame + 'px';
    }
    if(redata_height_based_on_width <= redata_height_frame) {
        redata_camera_video.style.height = redata_height_based_on_width + 'px';
    } else {
        redata_camera_video.style.height = redata_height_frame + 'px';
    }

    let translate_width_frame = translate_camera_frame.getBoundingClientRect().width;
    let translate_height_frame= translate_camera_frame.getBoundingClientRect().height;
    let translate_width_based_on_height = (translate_height_frame * 4) / 3;
    let translate_height_based_on_width = (translate_width_frame * 3) / 4;
    if(translate_width_based_on_height <= translate_width_frame) {
        translate_camera_video.style.width = translate_width_based_on_height + 'px';
    } else {
        translate_camera_video.style.width = translate_width_frame + 'px';
    }
    if(translate_height_based_on_width <= translate_height_frame) {
        translate_camera_video.style.height = translate_height_based_on_width + 'px';
    } else {
        translate_camera_video.style.height = translate_height_frame + 'px';
    }
});