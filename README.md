# EEG-Classifier-Softwate-Engineering
Repository for a system to classify EEG signals using Machine Learning techniques



El archivo `Dockerfile` configura un entorno basado en Python 3.11.9 con las dependencias necesarias. A continuación, se detallan los pasos para construir y ejecutar el contenedor usando Docker.

## Requisitos

- **Docker**: Necesitas tener Docker instalado en tu sistema. Puedes verificar la instalación ejecutando el siguiente comando en la terminal:

  _bash_
  docker --version
  _

### Instalación de Docker

1. **Windows**:
   - Descarga el instalador desde [Docker Desktop para Windows](https://docs.docker.com/desktop/install/windows-install/).
   - Sigue las instrucciones del instalador y asegúrate de que Docker esté ejecutándose después de la instalación.

2. **MacOS**:
   - Descarga Docker Desktop desde [Docker Desktop para Mac](https://docs.docker.com/desktop/install/mac-install/).
   - Sigue las instrucciones de instalación y verifica que Docker esté en ejecución.

3. **Linux**:
   - Sigue las instrucciones específicas para tu distribución en la [documentación oficial de Docker para Linux](https://docs.docker.com/engine/install/).

## Ejecución con Docker

### 1. Clonar o descargar el repositorio

Descarga o clona el repositorio que contiene el `Dockerfile`, el script `run.bat`, y el código de la aplicación.

### 2. Ejecutar el contenedor con `run.bat`

Para iniciar la aplicación en un sistema Windows, ejecuta el archivo `run.bat`. Este script está configurado para:

1. Ejecutar el contenedor Docker basado en la imagen construida en el paso anterior.
2. Iniciar el archivo principal `app.py` de la aplicación dentro del contenedor.

## Notas adicionales

- **Compatibilidad**: Docker garantiza un entorno aislado, eliminando problemas de configuración y compatibilidad de dependencias.
- **Solución de problemas**: Si encuentras problemas al ejecutar la aplicación, asegúrate de que Docker esté actualizado y funcionando correctamente en tu sistema.
