<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]

<br />
<!-- <div align="center"> -->

<!-- ABOUT THE PROJECT -->
## Acerca del Proyecto

Simulación del equilibrio hidrostático en el Sol como parte del curso de Hidrodinámica y Magnetohidrodinámica en la Universidad Autónoma de Nuevo León para la Maestría en Astrofísica Planetaria y Tecnologías Afines.

<!-- GETTING STARTED -->
## Corriendo el Programa

### Pre-requisitos

#### CMake
Este proyecto está hecho usando [CMake](https://cmake.org), un conjunto de herramientas para facilitar la compilación de código (en este caso en C++) de manera independiente de plataforma. Para instalarlo siga los siguientes pasos:

Linux
```
sudo apt-get install cmake
```

MacOS (a través de [homebrew](https://brew.sh))
```
brew install cmake
```

Windows: descarga el archivo binario correspondiente a tu sistema operativo de la [página de descargas de CMake](https://cmake.org/download/).

#### Compilador C++
El compilador usado durante el desarrollo de este programa ha sido el compilador [GNU de C++](https://gcc.gnu.org), por lo cual es el recomendado al compilar en Linux.

Linux
```
 sudo apt-get install gcc
```

En MacOS puede intentar usar el compilador de `clang++` que viene con las herramientas de [XCode para la terminal](https://mac.install.guide/commandlinetools/index.html) si ya lo tiene instalado. Si no, puede intentar [instalar gcc](https://discussions.apple.com/thread/8336714) en su Mac.

#### Make
Para compilar el código a un archivo binario ejecutable necesita una herramienta de compilación; esta tarea no está hecha por `cmake`. 

Linux
```
sudo apt-get install make
```

MacOS
```
xcode-select --install
```

O como alternativa a través de `homebrew`:
```
brew install make
```

<!-- USAGE EXAMPLES -->
## Usar el Código
### Compilación

Asegura de estar en la carpeta de más alto nivel del proyecto (`EquilibrioHidrostatico`)
```
mkdir build && cd build 
cmake ..
make
```

### Correr el Código
Una vez que el archivo binario haya sido generado lo podrá correr dentro de la carpeta `build` que creamos en el paso anterior:
```
./EquilibrioHidrostatico
```

En la consola podrá ver la ubicación del archivo donde se guardó los resultados de la integración numérica.

<!-- CONTACT -->
## Contact

Ramón Caballero Villegas - ramon.caballerov@uanl.edu.mx

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/KnightIV/EquilibrioHidrostatico/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/KnightIV/EquilibrioHidrostatico/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/KnightIV/EquilibrioHidrostatico/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/KnightIV/EquilibrioHidrostatico/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
