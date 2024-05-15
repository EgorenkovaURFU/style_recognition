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
    }
}
