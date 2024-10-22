FROM php:7.4.0-fpm

RUN docker-php-ext-install pdo pdo_mysql
