import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss']
})
export class ResultComponent implements OnInit {

  getJsonValue: any;
  postJsonValue: any;
  courses: any;
  list: any;

  constructor(
    private route: ActivatedRoute,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe((params: any) => {
      this.postJsonValue = params;
    })
    this.postMethod();
    // this.displayData();
  }
  
  postMethod() {
    this.http.post('http://127.0.0.1:5000/recommend', this.postJsonValue).subscribe((receivedData) => {
      this.getJsonValue = receivedData;
      this.courses = receivedData;
    });
    setTimeout(() => {
      this.displayData()
    }, 100)
  }
  
  displayData() {
    this.list = document.getElementById("recommendations");
    this.courses.forEach((item: any) => {
      let li = document.createElement("li");
      li.innerText = String(item);
      this.list.appendChild(li);
    });
  }

}
