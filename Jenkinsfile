pipeline {
    agent any
    environment {
        JENKINS_HOME = "$JENKINS_HOME"
        BUILD = "${JENKINS_HOME}/workspace/style_recognition"
        DOCKER_IMAGE_NAME = 'style_recognition'
    }
    stages {
        stage('Install DVC') {
            steps {
                sh 'pip install dvc'
            }
        }
        stage('Init DVC') {
            steps {
                sh 'dvc init'
            }
        }
        stage('Build Docker image'){
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_NAME} .'
            }
        }
        
        stage( 'RUN Docker'){
            steps{
                sh 'docker run -d -p 8501:8501 --name style_recognition-app ${DOCKER_IMAGE_NAME}'
            }
        }

        stage( 'Installation modules'){
            steps{
                sh 'pip install pytest'
                sh 'pip install streamlit'
               
            }
        }
        
        stage( 'RUN Test'){
            steps{
                sh 'python3 test/test_main.py'
            }
        }
        stage( 'DVC'){
            steps{
                sh 'dvc add result'
                sh 'dvc commit -m "Создание DVC-метки"'
                sh 'dvc push'
            }
        }
    }
}
