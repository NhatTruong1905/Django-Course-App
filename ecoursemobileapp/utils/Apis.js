import axios from "axios";


const BASE_URL = 'https://thanhduong.pythonanywhere.com/';

export const endpoints = {
    'categories': '/categories/',
    'courses': '/courses/'
};

export default axios.create({
    baseURL: BASE_URL
});