import { Component } from '@angular/core';
import { AppService, Prediction } from './app.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'frontend';
  file: any;
  imageSrc: string;
  prediction: Prediction;
  constructor(private appService: AppService) {}

  onChange(event) {
    this.file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = e => this.imageSrc = reader.result as string;
    reader.readAsDataURL(this.file);
  }

  predict() {
    this.appService.predict(this.file).subscribe(data => {
      this.prediction = data;
    });
  }

}
