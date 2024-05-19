pipeline {
    agent any
    environment {
        JENKINS_HOME = "$JENKINS_HOME"
        BUILD = "${JENKINS_HOME}/workspace/style_recognition"
        DOCKER_IMAGE_NAME = 'style_recognition'
    }

    stages{
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


        stage( 'Installation of modules'){
            steps{
                sh 'pip3 install pytest'
                sh 'pip install streamlit'
                sh 'pip install dvc'
                sh 'pip install dvc-gdrive'
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
                sh 'dvc commit -m "результат"'
                sh 'dvc push '
            }
        }
    }
}
