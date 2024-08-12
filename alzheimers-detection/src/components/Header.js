import React from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import logo from '../assets/images/logo.png';
import './Header.css';

const Header = () => (
  <Navbar bg="light" expand="lg">
    <Container className="ml-3">
      <Navbar.Brand as={Link} to="/" className="d-flex align-items-center">
        <img
          src={logo}
          className="navbar-logo"
          alt="AlzGuard logo"
        />
        AlzGuard
      </Navbar.Brand>
      <Navbar.Toggle aria-controls="basic-navbar-nav" />
      <div>
        <Navbar.Collapse className="basic-navbar-nav">
          <Nav className="me-auto">
            <Nav.Link as={Link} to="/">Home</Nav.Link>
            <Nav.Link as={Link} to="/about">About</Nav.Link>
            <Nav.Link as={Link} to="/how-it-works">How It Works</Nav.Link>
            <Nav.Link as={Link} to="/contact">Contact</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </div>
    </Container>
  </Navbar>
);

export default Header;
