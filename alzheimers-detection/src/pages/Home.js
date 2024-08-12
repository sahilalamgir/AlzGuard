import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Image } from 'react-bootstrap';
import axios from 'axios';
import './Home.css';

const Home = () => {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [ethnicity, setEthnicity] = useState('');
  const [diabetes, setDiabetes] = useState('');
  const [cholesterolHDL, setCholesterolHDL] = useState('');
  const [MMSE, setMMSE] = useState('');
  const [functionalAssessment, setFunctionalAssessment] = useState('');
  const [memoryComplaints, setMemoryComplaints] = useState('');
  const [behavioralProblems, setBehavioralProblems] = useState('');
  const [ADL, setADL] = useState('');
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(URL.createObjectURL(e.target.files[0]));
      setImageFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('ethnicity', ethnicity);
    formData.append('diabetes', diabetes);
    formData.append('cholesterol_HDL', cholesterolHDL);
    formData.append('MMSE', MMSE);
    formData.append('functional_assessment', functionalAssessment);
    formData.append('memory_complaints', memoryComplaints);
    formData.append('behavioral_problems', behavioralProblems);
    formData.append('ADL', ADL);
    formData.append('mri_image', imageFile);

    axios.post('http://localhost:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    .then(response => {
      setResult(response.data);
    })
    .catch(error => {
      console.error('There was an error!', error);
    });
  };

  return (
    <div className="home">
      <Container>
        <Row className="hero mb-4 text-center">
          <Col className='my-4'>
            <h1>Alzheimer's Detection</h1>
            <p>Utilizing AI to detect Alzheimer's early through MRI scans.</p>
          </Col>
        </Row>
        <Row className="form-section">
          <Col md={6} className="form-col">
            <Form onSubmit={handleSubmit}>
              <Form.Group className="mb-3" controlId="ethnicity">
                <Form.Label>Ethnicity</Form.Label>
                <Form.Control 
                  as="select" 
                  value={ethnicity} 
                  onChange={(e) => setEthnicity(e.target.value)}
                >
                  <option value="">Select Ethnicity</option>
                  <option value="0">Caucasian</option>
                  <option value="1">African American</option>
                  <option value="2">Asian</option>
                  <option value="3">Other</option>
                </Form.Control>
              </Form.Group>
              <Form.Group className="mb-3" controlId="diabetes">
                <Form.Label>Diabetes</Form.Label>
                <Form.Control 
                  as="select" 
                  value={diabetes} 
                  onChange={(e) => setDiabetes(e.target.value)}
                >
                  <option value="">Select Diabetes Status</option>
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </Form.Control>
              </Form.Group>
              <Form.Group className="mb-3" controlId="cholesterolHDL">
                <Form.Label>Cholesterol HDL</Form.Label>
                <Form.Control 
                  type="number" 
                  placeholder="Enter cholesterol HDL" 
                  value={cholesterolHDL}
                  onChange={(e) => setCholesterolHDL(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="MMSE">
                <Form.Label>MMSE</Form.Label>
                <Form.Control 
                  type="number" 
                  placeholder="Enter MMSE score" 
                  value={MMSE}
                  onChange={(e) => setMMSE(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="functionalAssessment">
                <Form.Label>Functional Assessment</Form.Label>
                <Form.Control 
                  type="text" 
                  placeholder="Enter functional assessment" 
                  value={functionalAssessment}
                  onChange={(e) => setFunctionalAssessment(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="memoryComplaints">
                <Form.Label>Memory Complaints</Form.Label>
                <Form.Control 
                  type="text" 
                  placeholder="Enter memory complaints" 
                  value={memoryComplaints}
                  onChange={(e) => setMemoryComplaints(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="behavioralProblems">
                <Form.Label>Behavioral Problems</Form.Label>
                <Form.Control 
                  type="text" 
                  placeholder="Enter behavioral problems" 
                  value={behavioralProblems}
                  onChange={(e) => setBehavioralProblems(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3" controlId="ADL">
                <Form.Label>ADL</Form.Label>
                <Form.Control 
                  type="text" 
                  placeholder="Enter ADL" 
                  value={ADL}
                  onChange={(e) => setADL(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-4" controlId="mriUpload">
                <Form.Label>MRI Scan Upload</Form.Label>
                <Form.Control type="file" accept="image/*" onChange={handleImageChange} />
              </Form.Group>
              <Button variant="primary" type="submit" className="w-100 mt-2">
                Get Results
              </Button>
            </Form>
          </Col>
          <Col md={6} className='img-col'>
            <div className="img-wrapper">
              {image ? <Image className='img-place' src={image} alt="Uploaded MRI" fluid /> : <div className="placeholder">Your image will appear here</div>}
            </div>
          </Col>
        </Row>
        {result && (
          <Row className="result-section mt-4 text-center">
            <Col>
              <h3>Prediction Result</h3>
              <p>Prediction: {result.prediction}</p>
              <p>Confidence: {result.confidence}</p>
            </Col>
          </Row>
        )}
      </Container>
    </div>
  );
};

export default Home;
