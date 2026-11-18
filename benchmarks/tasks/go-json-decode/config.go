package main

import "encoding/json"

type Config struct {
	Port int `json:"port_number"` // BUG: wrong tag, so "port" is ignored
}

func Decode(data []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}
	return &c, nil
}
