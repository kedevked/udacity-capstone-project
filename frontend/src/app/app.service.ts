import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class AppService {

  constructor(private httpClient: HttpClient) { }
/**
 *
 * @param img image file
 */
  predict(img) {
    const formData = new FormData();
    formData.append('file', img);
     return this.httpClient.post<Prediction>('http://localhost:5000/predict', formData);
  }
}

export interface Prediction {
  type: Type;
  breed?: string;
  error?: string;
}

export enum Type {
  HUMAN = 'human',
  DOG = 'dog',
  OTHER = 'other'
}
