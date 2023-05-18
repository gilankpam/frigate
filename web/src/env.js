export const ENV = import.meta.env.MODE;
export const API_HOST = ENV === 'production' ? '' : 'http://orangepi5.local:5000/';
